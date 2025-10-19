#!/usr/bin/env python3
"""
analyze_repo_llm.py

用途：完全使用 LLM（OpenAI）来产出仓库主要功能报告（JSON）。

用法：
  export OPENAI_API_KEY="sk-..."
  python analyze_repo_llm.py /path/to/repo.zip
  或
  python analyze_repo_llm.py /path/to/repo_dir
"""

import os
import sys
import zipfile
import tempfile
import shutil
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    import openai
except Exception as e:
    raise RuntimeError("请先安装 openai 库：pip install openai") from e

# ---------------- config ----------------
MODEL = os.environ.get("ANALYZER_MODEL", "gpt-4o-mini")  # 可替换为 gpt-4o / gpt-4 等
CHUNK_TOKENS = int(os.environ.get("ANALYZER_CHUNK_TOKENS", 1800))  # 每个 chunk 目标 token
EST_CHARS_PER_TOKEN = 4  # 经验值：约 4 字符/ token（中文略差）
CHUNK_CHARS = CHUNK_TOKENS * EST_CHARS_PER_TOKEN
MAX_FILE_BYTES = 2 * 1024 * 1024  # 超过 2MB 的文件将被部分读取或分块
SKIP_DIRS = {"src"}
TEXT_EXT = {".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".h", ".html", ".css"}

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("需要设置 OPENAI_API_KEY 环境变量（例如 export OPENAI_API_KEY=sk-...）")
openai.api_key = OPENAI_KEY

# ---------------- helpers ----------------

def safe_extract_zip(src: str, dst: str):
    with zipfile.ZipFile(src, "r") as z:
        for member in z.infolist():
            nm = os.path.normpath(member.filename)
            if nm.startswith("..") or os.path.isabs(nm):
                continue
            target = os.path.join(dst, nm)
            target_dir = os.path.dirname(target)
            os.makedirs(target_dir, exist_ok=True)
            if member.is_dir():
                os.makedirs(target, exist_ok=True)
            else:
                with z.open(member) as reader, open(target, "wb") as writer:
                    shutil.copyfileobj(reader, writer)

def is_text_file(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in TEXT_EXT:
        return True
    # 小文件尝试读取前 1KB 判断是否文本
    try:
        with open(path, "rb") as f:
            h = f.read(1024)
            if not h:
                return False
            # 如果包含大量 0 bytes，疑似二进制
            if b"\x00" in h:
                return False
            # otherwise assume text
            return True
    except Exception:
        return False

def iter_source_files(root: Path) -> List[Path]:
    out = []
    for p in root.rglob("*"):
        if p.is_file():
            parts = set(p.parts)
            if parts & SKIP_DIRS:
                # continue
                if is_text_file(p):
                    print(p)
                    out.append(p)
    return out

def read_file_safe(path: Path, max_bytes: int = MAX_FILE_BYTES) -> str:
    # 以 utf-8 读取，回退到 latin1 如果失败
    try:
        size = path.stat().st_size
        if size > max_bytes:
            # 只读前 max_bytes 字节以节省费用/时间（后面会按 chunk 分）
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(max_bytes)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        try:
            with open(path, "r", encoding="latin1", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS) -> List[str]:
    # 按行分割并拼接到 chunk_chars 限制，避免截断句子
    lines = text.splitlines(keepends=True)
    chunks = []
    cur = []
    cur_len = 0
    for ln in lines:
        ln_len = len(ln)
        if cur_len + ln_len > chunk_chars and cur:
            chunks.append("".join(cur))
            cur = [ln]
            cur_len = ln_len
        else:
            cur.append(ln)
            cur_len += ln_len
    if cur:
        chunks.append("".join(cur))
    # 若某一行超长，进一步硬切
    fixed = []
    for c in chunks:
        if len(c) <= chunk_chars:
            fixed.append(c)
        else:
            # hard chop
            for i in range(0, len(c), chunk_chars):
                fixed.append(c[i:i+chunk_chars])
    return fixed

# ---------------- LLM interactions ----------------

def chat_completion(prompt_messages: List[Dict[str, str]], model: str = MODEL, max_tokens: int = 1500) -> str:
    # 简单封装 OpenAI ChatCompletion（同步）
    resp = openai.ChatCompletion.create(
        model=model,
        messages=prompt_messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp["choices"][0]["message"]["content"]

# prompt 模板：对 chunk 做摘要
FILE_CHUNK_PROMPT_SYSTEM = (
    "你是一个代码审查助理。目标是从给定的代码片段中抽取出："
    "（1）该片段所属文件（如果已知），（2）功能函数简单描述，"
    "（3）该片段中的关键函数名称（列表），（4）每个函数/API的代码的行范围如1-10，"
    "（5）若该片段看起来像配置/构建脚本或非业务代码请标记为 'non-business'。"
    "只返回 JSON 对象，不要额外的文字。"
)

FILE_CHUNK_PROMPT_USER = (
    "文件名: {filename}\n"
    "文件相对路径: {relpath}\n"
    "下面是文件内容片段（用三引号分隔）：\n\n\"\"\"\n{chunk}\n\"\"\"\n\n"
    "请按 system 的要求返回 JSON，字段建议： {{'filename', 'summary', 'keys', 'related_line', 'type'}}。"
)


# prompt 模板：聚合文件级摘要
AGGREGATE_PROMPT_SYSTEM = (
    "你是一个资深软件工程师，负责把一堆文件级摘要汇总成仓库的“主要功能报告”。"
    "输入是 JSON 列表，每项包含 file, summary, keys, related_features, type。"
    "请生成："
    " (1) concise_repo_summary: 2-4 行概述仓库做什么；"
    " (2) features: 列表，每个 函数 包含 file（相关文件）,name（函数名称）, description, line (引用若干行), confidence（高/中/低）；"
    " (3) suggested_entrypoints: 若能发现 README / package.json / main.py 等，请列出如何运行/测试的线索（可选）。"
    "只返回 JSON，严格按照结构。"
)

AGGREGATE_PROMPT_USER = (
    "请对下面 JSON 数组做汇总（JSON 数组）：\n\n{file_summaries}\n\n"
    "返回 JSON 对象：{{ 'concise_repo_summary', 'features', 'suggested_entrypoints' }}。"
)

# ---------------- main flow ----------------

def analyze_repo_with_llm(root_path: str, out_json: str = "repo_report.json") -> Dict[str, Any]:
    tmpdir = None
    root = Path(root_path)
    # 如果是 zip 解压
    if zipfile.is_zipfile(root_path):
        tmpdir = tempfile.mkdtemp(prefix="repo_unzip_")
        safe_extract_zip(root_path, tmpdir)
        root = Path(tmpdir)

    files = iter_source_files(root)
    # 读取并分片、对每个 chunk 询问 LLM
    file_summaries = []
    print(f"找到 {len(files)} 个候选文本文件，开始 LLM 摘要...")

    for fp in files:
        try:
            rel = str(fp.relative_to(root))
        except Exception:
            rel = fp.name
        text = read_file_safe(fp)
        if not text.strip():
            continue
        # 为了节省 token 先用文件内容开头作快速判定
        # 切片
        chunks = chunk_text(text)
        chunk_summaries = []
        for i, ch in enumerate(chunks):
            # prompt
            sys_msg = {"role": "system", "content": FILE_CHUNK_PROMPT_SYSTEM}
            user_msg = {"role": "user", "content": FILE_CHUNK_PROMPT_USER.format(filename=fp.name, relpath=rel, chunk=ch[:CHUNK_CHARS])}
            try:
                from openai import OpenAI

                client = OpenAI()  # 会自动读取 OPENAI_API_KEY 环境变量

                def chat_completion(messages, model="gpt-4o-mini", max_tokens=1500):
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.0,
                    )
                    # resp 是 pydantic model；取文本如下
                    return resp.choices[0].message.content
                out = chat_completion([sys_msg, user_msg], max_tokens=1000)
                print(out)
            except Exception as e:
                # 若调用失败，降级为空摘要并继续
                out = json.dumps({"filename": fp.name, "summary": "", "keys": [], "related_features": [], "type": "error", "error": str(e)}, ensure_ascii=False)
            # 尝试解析 JSON（LLM 可能返回多余文本）
            parsed = None
            try:
                parsed = json.loads(out)
            except Exception:
                # 尝试提取首个 JSON 对象
                m = re.search(r"(\{.*\})", out, flags=re.S)
                if m:
                    try:
                        parsed = json.loads(m.group(1))
                    except Exception:
                        parsed = {"filename": fp.name, "summary": out.strip(), "keys": [], "related_features": [], "type": "unknown"}
                else:
                    parsed = {"filename": fp.name, "summary": out.strip(), "keys": [], "related_features": [], "type": "unknown"}
            parsed.setdefault("filename", fp.name)
            parsed.setdefault("relpath", rel)
            parsed.setdefault("chunk_index", i)
            chunk_summaries.append(parsed)
        # 聚合 chunk summaries（把 summary 合并，keys 合并）
        combined_summary = {
            "file": rel,
            "filename": fp.name,
            "summary": " ".join([c.get("summary","") for c in chunk_summaries]).strip()[:2000],
            "keys": list({k for c in chunk_summaries for k in (c.get("keys") or [])}),
            "related_features": list({f for c in chunk_summaries for f in (c.get("related_features") or [])}),
            "types": list({c.get("type","unknown") for c in chunk_summaries}),
            "chunks": chunk_summaries
        }
        file_summaries.append(combined_summary)

    # 把文件级摘要交给 LLM 做全仓库聚合
    # 限制传输大小：将 file_summaries JSON 压缩成文本分段传输
    file_summaries_json = json.dumps(file_summaries, ensure_ascii=False)
    # 若过大，截取前 N 文件的摘要（一般已足够）
    truncated = False
    if len(file_summaries_json) > CHUNK_CHARS * 4:
        # 只保留前 300KB 的摘要作为输入（可配置）
        truncated = True
        approx_keep = int((CHUNK_CHARS * 4) / len(file_summaries_json) * len(file_summaries))
        approx_keep = max(5, approx_keep)
        file_summaries = file_summaries[:approx_keep]
        file_summaries_json = json.dumps(file_summaries, ensure_ascii=False)

    sys_msg = {"role": "system", "content": AGGREGATE_PROMPT_SYSTEM}
    user_msg = {"role": "user", "content": AGGREGATE_PROMPT_USER.format(file_summaries=file_summaries_json)}

    print("正在调用 LLM 聚合仓库级报告（可能需要几秒到几十秒）...")
    try:
        agg_out = chat_completion([sys_msg, user_msg], max_tokens=1500)
    except Exception as e:
        agg_out = json.dumps({"concise_repo_summary": "", "features": [], "suggested_entrypoints": [], "error": str(e)}, ensure_ascii=False)

    # 尝试解析
    try:
        report_obj = json.loads(agg_out)
    except Exception:
        m = re.search(r"(\{.*\})", agg_out, flags=re.S)
        if m:
            try:
                report_obj = json.loads(m.group(1))
            except Exception:
                report_obj = {"concise_repo_summary": agg_out.strip(), "features": [], "suggested_entrypoints": []}
        else:
            report_obj = {"concise_repo_summary": agg_out.strip(), "features": [], "suggested_entrypoints": []}

    final = {
        "repo_path": str(root),
        "truncated_file_summaries": truncated,
        "file_summaries_count": len(file_summaries),
        "file_summaries": file_summaries,
        "report": report_obj
    }

    # 写入文件
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # cleanup
    if tmpdir:
        shutil.rmtree(tmpdir)
    return final

# ---------------- CLI ----------------

def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_repo_llm.py /path/to/repo.zip_or_dir [out.json]")
        sys.exit(1)
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >=3 else "repo_report.json"
    if not os.path.exists(path):
        print("路径不存在：", path)
        sys.exit(1)
    print("开始分析：", path)
    res = analyze_repo_with_llm(path, out_json=out)
    print("完成，输出：", out)
    print("简短摘要：", res.get("report", {}).get("concise_repo_summary", "")[:500])

if __name__ == "__main__":
    main()
