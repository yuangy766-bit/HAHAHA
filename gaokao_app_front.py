# app.py
import os, json, math, io, csv
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------- 页面配置 --------------
st.set_page_config(page_title="高考数学能力画像（AI版）", layout="wide")
plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','Arial']
plt.rcParams['axes.unicode_minus'] = False

st.title("高考数学能力画像（AI 研究演示）")
st.caption("上传数据 → 字段映射 → 选择卷别（内置“近年题型/能力”先验，可调）→ 计算六维能力 → 图表展示 → AI 生成分析 → 与 AI 对话（含情绪安慰）")

# -------------- AI 调用封装（兼容“未配置密钥”）--------------
def llm_enabled():
    return bool(os.getenv("OPENAI_API_KEY","").strip())

from openai import OpenAI
import os

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    使用 gptsapi.net 网关的安全 LLM 调用。
    如果没有配置 Secrets，则自动返回启发式安慰/学习建议，不会导致应用崩溃。
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return heuristic_reply(user_prompt)

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.gptsapi.net/v1")
    model_id = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.25,
            max_tokens=800
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"(AI 调用失败：{e})\n改为启发式建议：\n" + heuristic_reply(user_prompt)

def heuristic_reply(text: str) -> str:
    """未配置 API 时的兜底建议（含情绪安慰）。"""
    t = (text or "").lower()
    tips = []
    if any(k in t for k in ["焦虑","紧张","压力","anxiety","anxious"]):
        tips.append("我理解你的紧张。先深呼吸 3 次，给自己 2 分钟缓冲。学习上从最低分维度开小步快跑，先拿到小胜利。")
    if any(k in t for k in ["沮丧","难过","低落","sad"]):
        tips.append("情绪低落很正常。今天先完成 10 道针对薄弱点的小题，做完就休息，第二天复盘错题。")
    if not tips:
        tips.append("建议从最低分维度入手：每天 10–15 题，次日复盘错题，并加做 1–2 道新情境（新题型）以巩固迁移。")
    return "\n".join(tips)

# -------------- 上传数据 --------------
st.header("① 上传 CSV/XLSX")
template = pd.DataFrame({
    "student_id":["S001","S001","S002"],
    "question_id":["Q0001","Q0002","Q0001"],
    "subject":["MATH","MATH","MATH"],
    "topic":["Functions","Probability","Geometry"],
    "qtype":["单选","填空","解答"],  # 可选：题型列
    "correct":[1,0,1],
    "time_spent_sec":[60,95,80],
    "attempts":[1,2,1],
    "question_level":[1,3,2],
    "is_new_type":[0,1,0],
})
st.download_button("下载数据模板（CSV）", data=template.to_csv(index=False).encode("utf-8"),
                   file_name="student_data_template.csv", mime="text/csv")

up = st.file_uploader("把 CSV 或 XLSX 拖到这里，或点击选择文件", type=["csv","xlsx"])
if up is None:
    st.info("尚未上传文件。可先下载模板，或上传你自己的数据表。")
    st.stop()

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

# -------------- 字段映射（适配不同表头） --------------
st.header("② 字段映射（自动建议 + 手动可改）")
target_fields = [
    "student_id","question_id","subject","topic","qtype",
    "correct","time_spent_sec","attempts","question_level","is_new_type"
]

def heuristic_map(columns):
    cols_lower = {c.lower(): c for c in columns}
    def find(*keys):
        for k in keys:
            for low, ori in cols_lower.items():
                if k in low:
                    return ori
        return None
    m = {}
    m["student_id"]     = find("student","sid","stu","学号")
    m["question_id"]    = find("question","qid","题号","小题")
    m["subject"]        = find("subject","course","科目")
    m["topic"]          = find("topic","knowledge","tag","chapter","unit","skill","知识点")
    m["qtype"]          = find("qtype","type","题型","题目类型")
    m["correct"]        = find("correct","is_correct","label","y","得分","是否正确")
    m["time_spent_sec"] = find("time","duration","seconds","sec","用时","耗时")
    m["attempts"]       = find("attempt","tries","trial","重做")
    m["question_level"] = find("level","difficulty","diff","hard","难度")
    m["is_new_type"]    = find("new","novel","innov","type_new","新题型")
    return {k:v for k,v in m.items() if v in columns}

heuristic = heuristic_map(list(raw.columns))
st.write("启发式建议：")
st.json(heuristic)

st.markdown("**最终映射（可下拉修改）**")
final_map = {}
cols = list(raw.columns)
for t in target_fields:
    default = heuristic.get(t) or None
    final_map[t] = st.selectbox(f"{t} ←", options=[None]+cols,
                                index=([None]+cols).index(default) if default in cols else 0,
                                key=f"map_{t}")

# 归一化
norm = pd.DataFrame()
for t in target_fields:
    if final_map[t] is not None:
        norm[t] = raw[final_map[t]]

# 最小必需列
min_required = ["student_id","question_id","correct"]
missing_min = [c for c in min_required if c not in norm.columns]
if missing_min:
    st.error(f"最少需要包含列：{missing_min}。请完成映射。")
    st.stop()

# 类型与范围
if "subject" in norm.columns:
    norm["subject"] = norm["subject"].astype(str).str.upper()
else:
    norm["subject"] = "MATH"

for c in ["correct","time_spent_sec","attempts","question_level","is_new_type"]:
    if c in norm.columns:
        norm[c] = pd.to_numeric(norm[c], errors="coerce")

if "question_level" in norm.columns:
    norm["question_level"] = norm["question_level"].clip(lower=1, upper=5)

math_df = norm[norm["subject"]=="MATH"].copy()
if math_df.empty:
    st.error("过滤后没有 MATH 记录。")
    st.stop()

# -------------- 卷别先验（近年趋势，默认可调） --------------
st.header("③ 选择卷别 & 先验（可手动微调）")

paper = st.selectbox("卷别", ["全国一卷（I）","全国二卷（乙）","新高考一卷（新课标I）"], index=0)
st.caption("说明：默认题型结构参考近年公开评析。支持侧栏微调以贴合你的样本。")

# 题型结构先验（比例）；乙卷常见“8单选+3多选+3填空+5解答”的结构在近年连续出现
# 参考：2025全国二卷“8+3+3+5”延续2024结构（来源：鸡西教育局网站评析）；近年全国卷强调基础性与核心素养考查（中国教育在线/新华社/南方网等评析）
qtype_prior = {
    "全国一卷（I）":      {"单选":0.40, "多选":0.10, "填空":0.17, "解答":0.33},
    "全国二卷（乙）":      {"单选":0.42, "多选":0.16, "填空":0.16, "解答":0.26},  # “8+3+3+5”导出的比率近似
    "新高考一卷（新课标I）": {"单选":0.40, "多选":0.10, "填空":0.17, "解答":0.33},
}

# 能力维度先验（结合“抽象/推理/建模/直观/运算/数据分析/创新”等核心素养导向）
ability_names = ["知识理解力","逻辑思维力","创造策略力","表达规范性","时间自控力","情绪稳定性"]
# 各题型对能力的典型权重（可调）
ability_by_qtype = {
    "单选": {"知识理解力":0.35,"逻辑思维力":0.25,"创造策略力":0.10,"表达规范性":0.10,"时间自控力":0.15,"情绪稳定性":0.05},
    "多选": {"知识理解力":0.25,"逻辑思维力":0.30,"创造策略力":0.20,"表达规范性":0.10,"时间自控力":0.10,"情绪稳定性":0.05},
    "填空": {"知识理解力":0.30,"逻辑思维力":0.25,"创造策略力":0.15,"表达规范性":0.05,"时间自控力":0.20,"情绪稳定性":0.05},
    "解答": {"知识理解力":0.20,"逻辑思维力":0.30,"创造策略力":0.20,"表达规范性":0.20,"时间自控力":0.05,"情绪稳定性":0.05},
}

# 侧栏允许覆盖先验
st.sidebar.header("题型分布权重（按卷别微调）")
qtw = qtype_prior[paper].copy()
for k in list(qtw.keys()):
    qtw[k] = st.sidebar.slider(f"{k} 占比(%)", 0, 100, int(round(qtw[k]*100)), 1)/100.0
# 归一化
s = sum(qtw.values())
if s>0: qtw = {k:v/s for k,v in qtw.items()}

st.sidebar.header("题型→能力 映射（可微调）")
ab_map = {}
for qt in ["单选","多选","填空","解答"]:
    ab_map[qt] = {}
    for ab in ability_names:
        default = ability_by_qtype[qt][ab]
        ab_map[qt][ab] = st.sidebar.slider(f"{qt}→{ab}(%)", 0, 100, int(default*100), 5)/100.0

# -------------- 评分权重 + 学生选择 --------------
st.header("④ 评分权重与学生选择")
W = {
    "知识理解力": st.slider("知识理解力 权重(%)", 0,100,25,1),
    "逻辑思维力": st.slider("逻辑思维力 权重(%)", 0,100,20,1),
    "创造策略力": st.slider("创造策略力 权重(%)", 0,100,15,1),
    "表达规范性": st.slider("表达规范性 权重(%)", 0,100,15,1),
    "时间自控力": st.slider("时间自控力 权重(%)", 0,100,15,1),
    "情绪稳定性": st.slider("情绪稳定性 权重(%)", 0,100,10,1),
}
W_sum = sum(W.values())
st.caption(f"当前权重总和：{W_sum}（计算总分时会自动归一化）")

students = sorted(math_df["student_id"].dropna().astype(str).unique())
sid = st.selectbox("选择学生", students, index=0)
sdf = math_df[math_df["student_id"].astype(str)==sid].copy()

# -------------- 能力打分函数（融合题型、难度、用时、尝试次数等） --------------
def score_by_ability(group: pd.DataFrame) -> dict:
    g = group.copy()
    # 题型列
    if "qtype" not in g.columns:
        g["qtype"] = "解答"  # 若无题型列，默认按“解答”处理
    # 基础清洗
    for c in ["correct","time_spent_sec","attempts","question_level","is_new_type"]:
        if c in g.columns: g[c] = pd.to_numeric(g[c], errors="coerce")
    # 基础组件
    # 正确率（按题型）
    type_acc = {}
    for qt in ["单选","多选","填空","解答"]:
        sub = g[g["qtype"].astype(str)==qt]
        type_acc[qt] = sub["correct"].mean() if len(sub)>0 else np.nan
    # 用时效率（按难度基线）
    def time_eff(x: pd.Series, lv: pd.Series):
        baseline = {1:60, 2:90, 3:120, 4:135, 5:150}
        ideal = sum(baseline.get(int(a),90) for a in lv.dropna().astype(int)) or 0
        actual = x.replace(0, np.nan).sum()
        if ideal<=0: return np.nan
        if actual<=0: return 1.0
        return max(0.0, min(1.2, ideal/actual))  # 封顶稍>1
    te = time_eff(g.get("time_spent_sec", pd.Series([])), g.get("question_level", pd.Series([])))

    # 维度合成：将“题型正确率”×（题型→能力映射）×（卷别题型占比）加权求和
    ab_raw = {ab:0.0 for ab in ability_names}
    weight_sum = {ab:0.0 for ab in ability_names}
    for qt, acc in type_acc.items():
        if np.isnan(acc): 
            continue
        for ab in ability_names:
            w = ab_map.get(qt,{}).get(ab,0.0) * qtw.get(qt,0.0)
            ab_raw[ab] += acc * w
            weight_sum[ab] += w
    # 转百分制 + 融入“尝试次数”“新题型”与“用时效率”等修正
    out = {}
    for ab in ability_names:
        base = (ab_raw[ab]/weight_sum[ab]*100.0) if weight_sum[ab]>0 else np.nan
        if np.isnan(base): base = (g["correct"].mean()*100.0) if "correct" in g else 0.0

        # 表达规范性：一次命中率（答对且 attempts==1）/总答对
        if ab=="表达规范性" and "attempts" in g.columns and "correct" in g.columns:
            total_corr = int((g["correct"]==1).sum())
            one_try = int(((g["correct"]==1) & (g["attempts"]==1)).sum())
            exp_bonus = (one_try/total_corr*100.0) if total_corr>0 else 0.0
            base = 0.6*base + 0.4*exp_bonus

        # 时间自控力：引入用时效率 te
        if ab=="时间自控力" and not pd.isna(te):
            base = min(100.0, base * (0.5 + 0.5*te))

        # 创造策略力：新题型正确率加权提升
        if ab=="创造策略力" and "is_new_type" in g.columns and "correct" in g.columns:
            new_mask = g["is_new_type"]==1
            if new_mask.any():
                new_acc = g.loc[new_mask,"correct"].mean()*100.0
                base = 0.6*base + 0.4*new_acc

        # 情绪稳定性：最长连续错误串越长扣分越多
        if ab=="情绪稳定性" and "correct" in g.columns:
            order = g.copy()
            if "question_id" in order.columns:
                try:
                    order["ord"] = order["question_id"].astype(str).str.extract(r"(\d+)").astype(float)
                    order = order.sort_values("ord")
                except:
                    order = order.reset_index(drop=True)
            else:
                order = order.reset_index(drop=True)
            longest = cur=0
            for c in order["correct"]:
                if int(c)==0: cur += 1;  longest = max(longest,cur)
                else: cur = 0
            emo = 100.0 if longest<=1 else max(0.0, 100.0-(longest-1)*20.0)
            base = 0.5*base + 0.5*emo

        out[ab] = float(max(0.0, min(100.0, base)))
    return out

scores = score_by_ability(sdf)
scores_int = {k:int(round(v)) for k,v in scores.items()}
overall = (sum(scores[k]*W[k] for k in scores) / (W_sum if W_sum>0 else 1.0)) if W_sum>0 else 0.0
overall_int = int(round(overall))

# -------------- 展示：表格 + 雷达图 + 分布图 --------------
left, right = st.columns([2,1])
with left:
    st.subheader(f"学生 {sid} 的六维能力评分")
    st.write(pd.Series({**scores_int, "总分": overall_int}).to_frame("得分"))

# 雷达图（单图）
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

st.subheader("其他图表（可用于报告）")
c3, c4 = st.columns(2)
with c3:
    if "question_level" in sdf.columns:
        fig2 = plt.figure(figsize=(5,4))
        sdf["question_level"].value_counts().sort_index().plot(kind="bar")
        plt.title("题目难度分布")
        plt.xlabel("难度"); plt.ylabel("数量")
        st.pyplot(fig2, use_container_width=True)
with c4:
    if "qtype" in sdf.columns:
        fig3 = plt.figure(figsize=(5,4))
        sdf["qtype"].astype(str).value_counts().reindex(["单选","多选","填空","解答"]).dropna().plot(kind="bar")
        plt.title("题型分布（样本）")
        plt.xlabel("题型"); plt.ylabel("数量")
        st.pyplot(fig3, use_container_width=True)

# -------------- AI 生成分析（中文） --------------
st.header("⑤ AI 生成分析（可选）")
st.caption("在 Streamlit Secrets 设置 OPENAI_API_KEY（可选 OPENAI_BASE_URL、OPENAI_MODEL）以启用模型；否则使用启发式建议。")
default_q = "请解释该学生的薄弱维度，并给出基于近年全国卷题型与能力考查趋势的 7 天训练计划（含每日题量、题型、知识点与时间分配建议）。"
user_q = st.text_area("向 AI 提问（可改写）", value=default_q, height=120)

if st.button("让 AI 分析这个学生"):
    ctx = {
        "student_id": sid,
        "paper": paper,
        "scores": scores_int,
        "overall": overall_int,
        "weights": W,
        "qtype_prior": qtw,
        "qtype_ability_map": ab_map,
        "row_count": int(len(sdf)),
        "columns": list(sdf.columns),
        "mapping": final_map,
        "note": "结合近年：基础性+核心素养（抽象/推理/建模/直观/运算/数据分析/创新）导向；乙卷常见8+3+3+5结构等。",
    }
    sys = "你是熟悉中国高考数学的智能助教，能把“题型结构/能力导向/数据表现”联系起来，给出清晰可执行的学习/训练建议。回答用中文。"
    prompt = f"上下文：{json.dumps(ctx, ensure_ascii=False)}\n\n问题：{user_q}"
    reply = call_llm(sys, prompt)
    st.markdown("**AI 分析：**")
    st.write(reply)
    with open("ai_analysis_log.jsonl","a",encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(), "ctx": ctx, "q": user_q, "a": reply}, ensure_ascii=False)+"\n")
    st.caption("已记录到 ai_analysis_log.jsonl")

# -------------- 与 AI 聊天（学习/情绪都可） --------------
st.header("⑥ 与 AI 对话（学习&情绪）")
st.caption("自由讨论刷题策略、心态调整等。若未设置密钥，系统会用启发式安慰/建议。")
if "chat" not in st.session_state:
    st.session_state.chat = []

def render_chat():
    for role, content in st.session_state.chat:
        if role=="user":
            st.markdown(f"**你：**{content}")
        else:
            st.markdown(f"**AI：**{content}")

render_chat()
user_msg = st.text_input("对 AI 说点什么……（回车或点击发送）")
if st.button("发送") and user_msg:
    st.session_state.chat.append(("user", user_msg))
    sys = "你是善解人意的学习助教。若用户有负面情绪，先共情安慰，再给 1–2 条可执行的小建议；否则给学习策略。中文回答。"
    ai_ans = call_llm(sys, user_msg)
    st.session_state.chat.append(("ai", ai_ans))
    with open("chat_history.jsonl","a",encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(), "user": user_msg, "ai": ai_ans}, ensure_ascii=False)+"\n")
    st.experimental_rerun()

# -------------- 导出（报告用） --------------
st.header("⑦ 下载产物（便于写研究报告）")
st.download_button("下载归一化数据（CSV）", data=math_df.to_csv(index=False).encode("utf-8"),
                   file_name="normalized_math.csv")

def calc_all(df):
    out=[]
    for s in sorted(df["student_id"].dropna().astype(str).unique()):
        g = df[df["student_id"].astype(str)==s]
        sc = score_by_ability(g)
        sc["student_id"]=s
        sc["总分"] = (sum(sc[k]*W[k] for k in W.keys()) / (W_sum if W_sum>0 else 1.0)) if W_sum>0 else 0.0
        out.append(sc)
    return pd.DataFrame(out)

all_scores = calc_all(math_df)
st.download_button("下载全体评分（CSV）", data=all_scores.to_csv(index=False).encode("utf-8"),
                   file_name="all_scores.csv")

buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight")
st.download_button("下载当前雷达图（PNG）", data=buf.getvalue(), file_name=f"{sid}_radar.png", mime="image/png")


