
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from io import StringIO, BytesIO
from datetime import datetime

st.set_page_config(page_title="高考数学能力画像评估系统（演示版）", layout="wide")

plt.rcParams['font.sans-serif'] = ['SimHei','Microsoft YaHei','STHeiti','Arial']
plt.rcParams['axes.unicode_minus'] = False

st.title("🎒 高考数学能力画像评估系统（研究演示脚本）")

st.markdown("""
**如何使用**
1. 在下面的**上传区域**将你的学生答题数据 `.csv` 或 `.xlsx` 拖拽进来（或点击选择）。  
2. 右侧侧边栏可调节各维度权重；下方可选择学生、查看雷达图与结果表。  
3. 底部可进行“符合/不符合”的主观反馈，系统会记录并用于后续微调。
""")

# ==== 1) 上传区（中心区域，不在侧边栏） ====
st.header("📤 数据上传")
required_cols = ["student_id","question_id","subject","topic","correct","time_spent_sec","attempts","question_level","is_new_type"]
optional_cols = ["essay_len"]
all_cols = required_cols + optional_cols

# 提供模板下载
template_df = pd.DataFrame({
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
csv_bytes = template_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 下载CSV模板", data=csv_bytes, file_name="student_data_template.csv", mime="text/csv", help="下载后按模板列名填入你的学生作答数据")

uploaded = st.file_uploader("将 CSV 或 XLSX 拖拽到此处，或点击选择文件", type=["csv","xlsx"], accept_multiple_files=False)

df = None
source_note = ""

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
            source_note = f"已载入 CSV：**{uploaded.name}**"
        else:
            df = pd.read_excel(uploaded)
            source_note = f"已载入 Excel：**{uploaded.name}**"
    except Exception as e:
        st.error(f"❌ 文件读取失败：{e}")
        st.stop()
else:
    # 没上传就尝试读取本地同目录的 student_data.csv（便于演示）
    demo_path = "student_data.csv"
    if os.path.exists(demo_path):
        df = pd.read_csv(demo_path)
        source_note = f"已使用示例数据：**{demo_path}**（建议上传你自己的数据）"
    else:
        st.info("还没有上传数据。你可以先下载模板，填好后再上传；或将 `student_data.csv` 放在脚本同目录用于演示。")
        st.stop()

st.success(source_note)

# ==== 2) 数据校验与预览 ====
st.subheader("👀 数据预览与校验")

missing = [c for c in required_cols if c not in df.columns]
extra = [c for c in df.columns if c not in all_cols]

c1, c2 = st.columns([3,2], gap="large")
with c1:
    st.dataframe(df.head(20), use_container_width=True)
with c2:
    if missing:
        st.error(f"缺少必要列：{missing}")
    else:
        st.success("必要列 ✔ 已全部包含")
    if extra:
        st.warning(f"存在未使用的附加列（可保留）：{extra}")
    st.caption("必要列: " + ", ".join(required_cols) + "；可选列: " + ", ".join(optional_cols))

# 如果缺列，停止执行
if missing:
    st.stop()

# ==== 3) 仅保留数学学科 ====
if 'subject' in df.columns:
    df['subject'] = df['subject'].astype(str).str.upper()
    math_df = df[df['subject'] == 'MATH'].copy()
else:
    math_df = df.copy()

if math_df.empty:
    st.error("数据中未找到 subject 为 MATH 的记录，请检查上传文件。")
    st.stop()

# 基本类型与边界处理
for col in ["correct","time_spent_sec","attempts","question_level","is_new_type"]:
    if col in math_df.columns:
        math_df[col] = pd.to_numeric(math_df[col], errors="coerce")
math_df["question_level"] = math_df["question_level"].clip(lower=1, upper=5)

# ==== 4) 侧边栏参数 ====
st.sidebar.header("⚙️ 评分权重（可调）")
def wslider(key, default):
    return st.sidebar.slider(key, 0, 100, default, 1)

w = {
    "知识理解力": wslider("知识理解力 权重(%)", 25),
    "逻辑思维力": wslider("逻辑思维力 权重(%)", 20),
    "创造策略力": wslider("创造策略力 权重(%)", 15),
    "表达沟通力": wslider("表达沟通力 权重(%)", 15),
    "时间自控力": wslider("时间自控力 权重(%)", 15),
    "情绪稳定性": wslider("情绪稳定性 权重(%)", 10),
}
w_sum = sum(w.values())
st.sidebar.caption(f"当前权重合计：{w_sum}（计算时将自动归一化）")

# ==== 5) 学生选择 ====
students = sorted(math_df['student_id'].dropna().astype(str).unique())
sid = st.selectbox("选择学生", students, index=0)

sdf = math_df[math_df['student_id'].astype(str) == sid].copy()
if sdf.empty:
    st.error("所选学生没有记录。")
    st.stop()

# ==== 6) 评分函数 ====
def s_knowledge(g):
    mask = g['question_level'].isin([1,2])
    score = g.loc[mask, 'correct'].mean()*100 if mask.any() else g['correct'].mean()*100
    return 0 if pd.isna(score) else float(score)

def s_logic(g):
    mask = g['question_level']>=3
    score = g.loc[mask, 'correct'].mean()*100 if mask.any() else g['correct'].mean()*100
    return 0 if pd.isna(score) else float(score)

def s_creative(g):
    mask = g['is_new_type']==1
    if mask.any():
        score = g.loc[mask, 'correct'].mean()*100
    else:
        score = s_logic(g)
    return 0 if pd.isna(score) else float(score)

def s_expression(g):
    total_correct = (g['correct']==1).sum()
    if total_correct==0:
        return 0.0
    one_try = ((g['correct']==1) & (g['attempts']==1)).sum()
    return float(one_try/total_correct*100)

def s_time(g):
    if len(g)==0:
        return 0.0
    baseline = {1:60, 2:90, 3:120, 4:135, 5:150}
    ideal = sum(baseline.get(int(x), 90) for x in g['question_level'])
    actual = g['time_spent_sec'].replace(0, np.nan).sum()
    if actual<=0:
        return 100.0 if ideal>0 else 0.0
    score = min(100.0, max(0.0, ideal/actual*100.0))
    return float(score)

def s_emotion(g):
    if len(g)==0:
        return 0.0
    order = g.copy()
    if 'question_id' in order.columns:
        try:
            order['ord'] = order['question_id'].astype(str).str.extract(r'(\d+)').astype(float)
            order = order.sort_values('ord')
        except:
            order = order.reset_index(drop=True)
    else:
        order = order.reset_index(drop=True)
    longest = cur = 0
    for c in order['correct']:
        if int(c)==0:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    if longest<=1:
        return 100.0
    return float(max(0.0, 100.0 - (longest-1)*20.0))

# ==== 7) 计算分数 ====
scores = {
    "知识理解力": s_knowledge(sdf),
    "逻辑思维力": s_logic(sdf),
    "创造策略力": s_creative(sdf),
    "表达沟通力": s_expression(sdf),
    "时间自控力": s_time(sdf),
    "情绪稳定性": s_emotion(sdf),
}
scores_int = {k:int(round(v)) for k,v in scores.items()}
total = (sum(scores[k]*w[k] for k in scores) / (w_sum if w_sum>0 else 1.0)) if w_sum>0 else 0.0
total_int = int(round(total))

# ==== 8) 展示 ====
cA, cB = st.columns([2,1])
with cA:
    st.subheader(f"学生 {sid} 能力评分")
    st.write(pd.Series({**scores_int, "总分": total_int}).to_frame("得分"))

dims = list(scores_int.keys())
vals = list(scores_int.values())
angles = [n/float(len(dims))*2*math.pi for n in range(len(dims))]
angles += angles[:1]; radar_vals = vals + vals[:1]

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, radar_vals, linewidth=2)
ax.fill(angles, radar_vals, alpha=0.25)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(dims)
ax.set_ylim(0,100); ax.set_title("六维能力雷达图", pad=14)
with cB:
    st.pyplot(fig, use_container_width=True)

st.markdown("---")
st.subheader("🗣️ 结果反馈（人机交互）")
c1, c2 = st.columns(2)
fb = None
with c1:
    if st.button("✅ 符合实际"):
        fb = "Yes"
        st.success("已记录：符合实际")
with c2:
    if st.button("❌ 不符合实际"):
        fb = "No"
        st.warning("已记录：不符合实际")
reason = st.text_input("若不符合，请简单说明原因（可选，如“低估逻辑”）")

if fb is not None:
    log = "feedback_history.csv"
    import csv, datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "time": ts,
        "student_id": sid,
        "fit": fb,
        "reason": reason,
        **{f"w_{k}": v for k,v in w.items()}
    }
    write_header = not os.path.exists(log)
    with open(log, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    st.success("反馈与当前权重已写入 feedback_history.csv")
