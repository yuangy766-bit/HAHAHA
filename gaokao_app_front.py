# app.py
import os, json, math, csv, io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------- 页面配置 --------------
st.set_page_config(page_title="高考数学能力画像（AI版）", layout="wide")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

st.title("高考数学能力画像（AI 研究演示）")
st.caption("上传数据 → 字段映射 → 计算六维能力 → 图表展示 → AI 生成分析 → 与 AI 对话（含情绪安慰）")

# -------------- 工具函数：是否启用 LLM --------------
def llm_enabled():
    return bool(os.getenv("OPENAI_API_KEY", "").strip())

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    如配置 OPENAI_API_KEY 就调用 LLM，否则返回启发式文本。
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return heuristic_reply(user_prompt)

    try:
        import openai
        openai.api_key = api_key
        # 你也可换成其他兼容的模型，如 gpt-4o-mini / gpt-3.5-turbo 等
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        return resp.choices[0].message["content"].strip()
    except Exception as e:
        return f"(LLM 调用失败：{e})\n下面给出启发式建议：\n" + heuristic_reply(user_prompt)

def heuristic_reply(text: str) -> str:
    """
    无 API Key 时的启发式建议/安慰。
    """
    t = text.lower()
    tips = []
    # 简单情绪识别与安慰
    if any(k in t for k in ["anxious", "anxiety", "焦虑", "紧张", "担心", "压力"]):
        tips.append("我能理解你的紧张和压力。先深呼吸，给自己 2–3 分钟放松。学习上，我们从最薄弱的一维开始，逐步建立小胜利。")
    if any(k in t for k in ["sad", "沮丧", "难过", "低落"]):
        tips.append("感到沮丧是正常的。你已经迈出改善的第一步。我们把目标拆小：今天先完成 10 道针对薄弱点的小题即可。")
    if not tips:
        tips.append("建议先从最低分的能力维度着手，一天 10–15 题，次日回顾错题并做 1–2 个新情境。保持节奏即可看到提升。")
    return "\n".join(tips)

# -------------- 上传数据 --------------
st.header("① 上传 CSV/XLSX")
template = pd.DataFrame({
    "student_id": ["S001","S001","S002"],
    "question_id": ["Q0001","Q0002","Q0001"],
    "subject": ["MATH","MATH","MATH"],
    "topic": ["Functions","Probability","Geometry"],
    "correct": [1,0,1],
    "time_spent_sec": [60,95,80],
    "attempts": [1,2,1],
    "question_level": [1,3,2],
    "is_new_type": [0,1,0],
    "essay_len": [0,0,0],
})
st.download_button(
    "下载数据模板（CSV）",
    data=template.to_csv(index=False).encode("utf-8"),
    file_name="student_data_template.csv",
    mime="text/csv"
)

up = st.file_uploader("把 CSV 或 XLSX 拖到这里，或者点击选择文件", type=["csv","xlsx"])
if up is None:
    st.info("尚未上传文件。可先下载模板，或上传你自己的数据表。")
    st.stop()

# 读取数据
try:
    if up.name.lower().endswith(".csv"):
        raw = pd.read_csv(up)
    else:
        raw = pd.read_excel(up)
    st.success(f"已读取文件：{up.name}（{len(raw)} 行 × {len(raw.columns)} 列）")
except Exception as e:
    st.error(f"读取失败：{e}")
    st.stop()

st.subheader("数据预览（前 20 行）")
st.dataframe(raw.head(20), use_container_width=True)

# -------------- 字段映射（适配不同格式表头） --------------
st.header("② 字段映射（自动建议 + 手动可改）")
target_fields = [
    "student_id","question_id","subject","topic",
    "correct","time_spent_sec","attempts","question_level","is_new_type","essay_len"
]

# 启发式：从列名里猜测映射
def heuristic_map(columns):
    cols_lower = {c.lower(): c for c in columns}
    def find(*keys):
        for k in keys:
            for low, ori in cols_lower.items():
                if k in low:
                    return ori
        return None
    m = {}
    m["student_id"]      = find("student", "sid", "stu")
    m["question_id"]     = find("question", "qid", "q_id", "item", "problem")
    m["subject"]         = find("subject", "course")
    m["topic"]           = find("topic", "knowledge", "tag", "chapter", "unit", "skill")
    m["correct"]         = find("correct", "is_correct", "label", "y", "score")
    m["time_spent_sec"]  = find("time", "duration", "seconds", "sec", "cost")
    m["attempts"]        = find("attempt", "tries", "trial")
    m["question_level"]  = find("level", "difficulty", "diff", "hard")
    m["is_new_type"]     = find("new", "novel", "innov", "type_new")
    m["essay_len"]       = find("essay", "words", "len", "length", "text_len")
    return {k:v for k,v in m.items() if v in columns}

heuristic = heuristic_map(list(raw.columns))

# （可选）AI 列映射建议
ai_mapping = {}
if llm_enabled():
    try:
        import openai
        sys = "你负责把用户上传表头映射到目标字段名（高考数学能力评估用）。"
        user = {
            "columns": list(raw.columns),
            "target_fields": target_fields,
            "instructions": "仅返回 JSON 对象：键为目标字段，值为上传表中对应列名或 null。不要多余说明。"
        }
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":sys},
                {"role":"user","content":"映射任务：\n"+json.dumps(user, ensure_ascii=False)}
            ],
            temperature=0.0,
            max_tokens=400
        )
        ai_text = resp.choices[0].message["content"].strip()
        ai_mapping = json.loads(ai_text) if ai_text.startswith("{") else {}
    except Exception:
        ai_mapping = {}

col1, col2 = st.columns(2)
with col1:
    st.write("AI 建议（若未配置 API Key 此处为空）：")
    st.json(ai_mapping if ai_mapping else {"note":"无 AI 建议"})
with col2:
    st.write("启发式建议：")
    st.json(heuristic)

st.markdown("**最终映射（可下拉修改）**")
final_map = {}
cols = list(raw.columns)
for t in target_fields:
    default = ai_mapping.get(t) or heuristic.get(t) or None
    final_map[t] = st.selectbox(
        f"{t} ←",
        options=[None]+cols,
        index=([None]+cols).index(default) if default in cols else 0,
        key=f"map_{t}"
    )

# 归一化后的数据框
norm = pd.DataFrame()
for t in target_fields:
    if final_map[t] is not None:
        norm[t] = raw[final_map[t]]

# 最小必需列
min_required = ["student_id", "question_id", "correct"]
missing_min = [c for c in min_required if c not in norm.columns]
if missing_min:
    st.error(f"最少需要包含列：{missing_min}。请完成映射。")
    st.stop()

# 类型与范围
if "subject" in norm.columns:
    norm["subject"] = norm["subject"].astype(str).str.upper()
else:
    norm["subject"] = "MATH"

for c in ["correct","time_spent_sec","attempts","question_level","is_new_type","essay_len"]:
    if c in norm.columns:
        norm[c] = pd.to_numeric(norm[c], errors="coerce")

if "question_level" in norm.columns:
    norm["question_level"] = norm["question_level"].clip(lower=1, upper=5)

math_df = norm[norm["subject"]=="MATH"].copy()
if math_df.empty:
    st.error("过滤后没有 MATH 记录。")
    st.stop()

# -------------- 六维能力计算 --------------
st.header("③ 评分权重与学生选择")

W = {
    "知识理解力": st.slider("知识理解力 权重(%)", 0, 100, 25, 1),
    "逻辑思维力": st.slider("逻辑思维力 权重(%)", 0, 100, 20, 1),
    "创造策略力": st.slider("创造策略力 权重(%)", 0, 100, 15, 1),
    "表达沟通力": st.slider("表达沟通力 权重(%)", 0, 100, 15, 1),
    "时间自控力": st.slider("时间自控力 权重(%)", 0, 100, 15, 1),
    "情绪稳定性": st.slider("情绪稳定性 权重(%)", 0, 100, 10, 1),
}
W_sum = sum(W.values())
st.caption(f"当前权重总和：{W_sum}（计算总分时会自动归一化）")

students = sorted(math_df["student_id"].dropna().astype(str).unique())
sid = st.selectbox("选择学生", students, index=0)
sdf = math_df[math_df["student_id"].astype(str)==sid].copy()

def S_知识(g):
    if "question_level" in g.columns and "correct" in g.columns:
        mask = g["question_level"].isin([1,2])
        score = g.loc[mask,"correct"].mean()*100 if mask.any() else g["correct"].mean()*100
    else:
        score = g["correct"].mean()*100 if "correct" in g.columns else 0.0
    return 0.0 if pd.isna(score) else float(score)

def S_逻辑(g):
    if "question_level" in g.columns and "correct" in g.columns:
        mask = g["question_level"]>=3
        score = g.loc[mask,"correct"].mean()*100 if mask.any() else g["correct"].mean()*100
    else:
        score = g["correct"].mean()*100 if "correct" in g.columns else 0.0
    return 0.0 if pd.isna(score) else float(score)

def S_策略(g):
    if "is_new_type" in g.columns and "correct" in g.columns:
        mask = g["is_new_type"]==1
        score = g.loc[mask,"correct"].mean()*100 if mask.any() else S_逻辑(g)
    else:
        score = S_逻辑(g)
    return 0.0 if pd.isna(score) else float(score)

def S_表达(g):
    if "correct" not in g.columns or "attempts" not in g.columns:
        return 0.0
    total_correct = int((g["correct"]==1).sum())
    if total_correct==0: return 0.0
    one_try = int(((g["correct"]==1) & (g["attempts"]==1)).sum())
    return float(one_try/total_correct*100)

def S_时间(g):
    if "time_spent_sec" not in g.columns or "question_level" not in g.columns:
        return 0.0
    if len(g)==0: return 0.0
    baseline = {1:60, 2:90, 3:120, 4:135, 5:150}
    ideal = sum(baseline.get(int(x),90) for x in g["question_level"])
    actual = g["time_spent_sec"].replace(0, np.nan).sum()
    if actual<=0: return 100.0 if ideal>0 else 0.0
    score = min(100.0, max(0.0, ideal/actual*100.0))
    return float(score)

def S_情绪(g):
    if "correct" not in g.columns: return 0.0
    order = g.copy()
    if "question_id" in order.columns:
        try:
            order["ord"] = order["question_id"].astype(str).str.extract(r"(\d+)").astype(float)
            order = order.sort_values("ord")
        except:
            order = order.reset_index(drop=True)
    else:
        order = order.reset_index(drop=True)
    longest = cur = 0
    for c in order["correct"]:
        if int(c)==0:
            cur += 1;  longest = max(longest, cur)
        else:
            cur = 0
    if longest<=1: return 100.0
    return float(max(0.0, 100.0 - (longest-1)*20.0))

scores = {
    "知识理解力": S_知识(sdf),
    "逻辑思维力": S_逻辑(sdf),
    "创造策略力": S_策略(sdf),
    "表达沟通力": S_表达(sdf),
    "时间自控力": S_时间(sdf),
    "情绪稳定性": S_情绪(sdf),
}
scores_int = {k:int(round(v)) for k,v in scores.items()}
overall = (sum(scores[k]*W[k] for k in scores) / (W_sum if W_sum>0 else 1.0)) if W_sum>0 else 0.0
overall_int = int(round(overall))

left, right = st.columns([2,1])
with left:
    st.subheader(f"学生 {sid} 的六维能力评分")
    st.write(pd.Series({**scores_int, "总分": overall_int}).to_frame("得分"))

# 雷达图（单独一张图）
dims = list(scores_int.keys())
vals = list(scores_int.values())
angles = [n/float(len(dims))*2*math.pi for n in range(len(dims))]
angles += angles[:1]; radar_vals = vals + vals[:1]

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, radar_vals, linewidth=2)
ax.fill(angles, radar_vals, alpha=0.25)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(dims)
ax.set_ylim(0,100); ax.set_title("六维能力雷达图", pad=16)
with right:
    st.pyplot(fig, use_container_width=True)

# 其他图表（示例：难度分布、topic分布）
st.subheader("其他图表（可用于报告）")
c3, c4 = st.columns(2)
with c3:
    if "question_level" in sdf.columns:
        fig2 = plt.figure(figsize=(5,4))
        sdf["question_level"].value_counts().sort_index().plot(kind="bar")
        plt.title("题目难度分布")
        plt.xlabel("难度等级"); plt.ylabel("数量")
        st.pyplot(fig2, use_container_width=True)
with c4:
    if "topic" in sdf.columns:
        fig3 = plt.figure(figsize=(5,4))
        sdf["topic"].astype(str).value_counts().head(10).plot(kind="bar")
        plt.title("知识点（Top10）")
        plt.xlabel("topic"); plt.ylabel("数量")
        st.pyplot(fig3, use_container_width=True)

# -------------- AI 生成分析（文字解读） --------------
st.header("④ AI 生成分析（可选）")
st.caption("若在 Streamlit Cloud 的 Secrets 配置了 OPENAI_API_KEY，将生成更完整的自然语言解读；否则使用启发式建议。")

default_q = "请解释该学生的薄弱维度，并给出基于高考题型分布的7天训练计划（含每日题量、题型与知识点建议）。"
user_q = st.text_area("向 AI 提问（可改写问题）", value=default_q, height=120)

if st.button("让 AI 分析这个学生"):
    ctx = {
        "student_id": sid,
        "scores": scores_int,
        "overall": overall_int,
        "weights": W,
        "row_count": int(len(sdf)),
        "columns": list(sdf.columns),
        "mapping": final_map,
        "note": "可结合近十年全国卷（全国I/甲/乙/新高考I）题型与知识点分布进行建议。"
    }
    sys = "你是熟悉中国高考数学的智能助教，能将数据与高考题型分布关联，给出清晰、可执行的学习建议。回答用中文。"
    prompt = f"上下文：{json.dumps(ctx, ensure_ascii=False)}\n\n问题：{user_q}"
    reply = call_llm(sys, prompt)
    st.markdown("**AI 分析：**")
    st.write(reply)
    with open("ai_analysis_log.jsonl","a",encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(), "ctx": ctx, "q": user_q, "a": reply}, ensure_ascii=False) + "\n")
    st.caption("已记录到 ai_analysis_log.jsonl")

# -------------- 与 AI 聊天（含情绪安慰） --------------
st.header("⑤ 与 AI 对话")
st.caption("随便聊学习/规划/情绪等。若配置 OPENAI_API_KEY 将由模型回答，否则给出启发式安慰/建议。")

if "chat" not in st.session_state:
    st.session_state.chat = []

def render_chat():
    for role, content in st.session_state.chat:
        if role == "user":
            st.markdown(f"**你：**{content}")
        else:
            st.markdown(f"**AI：**{content}")

render_chat()
user_msg = st.text_input("对 AI 说点什么……（按 Enter 发送）")
if st.button("发送") or (user_msg and st.session_state.get("auto_send_once") is None):
    if user_msg:
        st.session_state.chat.append(("user", user_msg))
        sys = "你是善解人意的学习助教，回答要真诚、具体、有可执行性；如检测到负面情绪，请先共情和安慰，再给出简单行动建议。回答中文。"
        ai_ans = call_llm(sys, user_msg)
        st.session_state.chat.append(("ai", ai_ans))
        with open("chat_history.jsonl","a",encoding="utf-8") as f:
            f.write(json.dumps({"time": datetime.now().isoformat(), "user": user_msg, "ai": ai_ans}, ensure_ascii=False) + "\n")
        st.session_state.auto_send_once = True
        st.experimental_rerun()

# -------------- 导出（用于写报告） --------------
st.header("⑥ 下载产物（便于写研究报告）")
st.download_button("下载归一化数据（CSV）", data=math_df.to_csv(index=False).encode("utf-8"), file_name="normalized_math.csv")
# 全体学生评分
def calc_all(df):
    out = []
    students_all = sorted(df["student_id"].dropna().astype(str).unique())
    for s in students_all:
        g = df[df["student_id"].astype(str)==s]
        sc = {
            "student_id": s,
            "知识理解力": S_知识(g),
            "逻辑思维力": S_逻辑(g),
            "创造策略力": S_策略(g),
            "表达沟通力": S_表达(g),
            "时间自控力": S_时间(g),
            "情绪稳定性": S_情绪(g),
        }
        sc["总分"] = (sum(sc[k]*W[k] for k in W.keys()) / (W_sum if W_sum>0 else 1.0)) if W_sum>0 else 0.0
        out.append(sc)
    return pd.DataFrame(out)

all_scores = calc_all(math_df)
st.download_button("下载全体评分（CSV）", data=all_scores.to_csv(index=False).encode("utf-8"), file_name="all_scores.csv")

# 导出雷达图 PNG
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight")
st.download_button("下载当前雷达图（PNG）", data=buf.getvalue(), file_name=f"{sid}_radar.png", mime="image/png")

