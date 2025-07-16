
import joblib
import shap
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

model = joblib.load("MSM5.pkl")

feature_names_chinese = [
    "过去一年是否出现自杀行为或想法",
    "目前的睡眠情况",
    "过去一年是否存在颜值焦虑",
    "社会支持得分",
    "内化同性恋嫌恶得分"
]

options_dict = {
    "过去一年是否出现自杀行为或想法": [("否", 1), ("是，有自杀意念", 2), ("是，有自杀计划", 3),("是，自杀未遂", 4)],
    "目前的睡眠情况": [("非常好", 1), ("较好", 2), ("一般", 3), ("较差", 4), ("非常差", 5)],
    "过去一年是否存在颜值焦虑": [("否", 0), ("是", 1)],
}

internalized_homophobia_items = [
    "1.如有可能，我宁愿选择成为真正的异性恋",
    "2.如果我是异性恋，我也许会活得更开心",
    "3.即使能够改变，我也不愿意成为异性恋",
    "4.在体制内工作一定不能暴露自己的性取向",
    "5.在工作单位暴露自己的性取向是会影响自己职业的发展",
    "6.大多数情况下，我并不介意他人知道我的性取向",
    "7.我担心别人知道我的性取向会让我的家人丢脸",
    "8.我没法和自己的伴侣在公共场合做异性伴侣做的事情",
    "9.我听到别人提起同性恋这三个字就会让我感到很紧张",
    "10.大多数同性恋都孤独终老",
    "11.作为一个同性恋我没法完成传统的孝道，会让我觉得自己不孝"
]

rating_options = ["非常不同意", "不同意", "不确定", "同意", "非常同意"]

def calculate_social_support():
    st.subheader("社会支持评估")
    score = 0
    item_scores = []
    questions = [
        ("1.您有多少关系密切，可以得到支持和帮助的朋友？", 
         ["一个也没有", "1-2个", "3-5个", "6个或6个以上"]),
        ("2.近一年来您：",
         ["远离家人，且独居一室", "住处经常变动，多数时间和陌生人住在一起", 
          "和同学、同事或朋友住在一起", "和家人住在一起"]),
        ("3.您与邻居：",
         ["相互之间从不关心，只是点头之交", "遇到困难可能稍微关心",
          "有些邻居都很关心您", "大多数邻居都很关心您"]),
        ("4.您与同事：",
         ["相互之间从不关心，只是点头之交", "遇到困难可能稍微关心",
          "有些同事很关心您", "大多数同事都很关心您"])
    ]
    
    for i, (question, options) in enumerate(questions):
        cols = st.columns([4, 1])
        with cols[0]:
            selected = st.radio(question, options, index=1, key=f"q{i+1}")
        with cols[1]:
            item_score = options.index(selected) + 1
            item_scores.append(item_score)
            st.text(f"{item_score}分")
        score += item_score
    
    st.write("5.从家庭成员得到的支持和照顾")
    family_members = ["5.1夫妻（恋人）", "5.2父母", "5.3儿女", "5.4兄弟姐妹", "5.5其他成员（如嫂子）"]
    for i, member in enumerate(family_members):
        cols = st.columns([4, 1])
        with cols[0]:
            support = st.radio(
                f"{member}的支持程度",
                ["无", "极少", "一般", "全力支持"],
                index=2,
                key=f"family_support_{i}"
            )
        with cols[1]:
            item_score = ["无", "极少", "一般", "全力支持"].index(support) + 1
            item_scores.append(item_score)
            st.text(f"{item_score}分")
        score += item_score
    
    st.write("6.过去，在您遇到急难情况时，曾经得到的经济支持和解决实际问题的帮助的来源有：（多选题）")
    options6 = ["无任何来源（选择该选项后，本题不计分）", "配偶", "其他家人", "亲戚", "朋友", "同事", 
               "工作单位", "党团工会等官方或半官方组织", "宗教、社会团体等非官方组织", "其它"]
    selected6 = st.multiselect("请选择所有适用的选项", options6, key="q6")
    item_score = 0 if "无任何来源" in selected6 else len(selected6)
    item_scores.append(item_score)
    score += item_score
    
    st.write("7.过去，在您遇到急难情况时，曾经得到的安慰和关心的来源有：（多选题）")
    options7 = ["无任何来源（选择该选项后，本题不计分）", "配偶", "其他家人", "亲戚", "朋友", "同事", 
               "工作单位", "党团工会等官方或半官方组织", "宗教、社会团体等非官方组织", "其它"]
    selected7 = st.multiselect("请选择所有适用的选项", options7, key="q7")
    item_score = 0 if "无任何来源" in selected7 else len(selected7)
    item_scores.append(item_score)
    score += item_score
    
    questions_8to10 = [
        ("8.您遇到烦恼时的倾诉方式：",
         ["从不向任何人诉述", "只向关系极为密切的1-2个人诉述",
          "如果朋友主动询问您会说出来", "主动诉述自己的烦恼，以获得支持和理解"]),
        ("9.您遇到烦恼时的求助方式：",
         ["只靠自己，不接受别人帮助", "很少请求别人帮助",
          "有时请求别人帮助", "有困难时经常向家人、亲友、组织求援"]),
        ("10.对于团体组织活动，您：",
         ["从不参加", "偶尔参加", "经常参加", "主动参加并积极活动"])
    ]
    
    for i, (question, options) in enumerate(questions_8to10):
        cols = st.columns([4, 1])
        with cols[0]:
            selected = st.radio(question, options, index=1, key=f"q{i+8}")
        with cols[1]:
            item_score = options.index(selected) + 1
            item_scores.append(item_score)
            st.text(f"{item_score}分")
        score += item_score
    
    return score, item_scores

def main():
    st.title("男男性行为者（MSM）抑郁风险预测模型")
    
    with st.form("input_form"):
        st.subheader("基本信息评估")
        inputs = {}
        for feature in feature_names_chinese[:3]:
            options = [option[0] for option in options_dict[feature]]
            inputs[feature] = st.selectbox(f"{feature}", options)
        
        social_support_score, support_item_scores = calculate_social_support()
        inputs["社会支持得分"] = social_support_score
        
        st.subheader("内化同性恋嫌恶评估")
        item_scores = []
        
        for i, item in enumerate(internalized_homophobia_items):
            cols = st.columns([4, 1])
            with cols[0]:
                rating = st.radio(
                    f"{i+1}. {item}",
                    rating_options,
                    index=2,
                    horizontal=True,
                    key=f"item_{i}"
                )
            with cols[1]:
                score = rating_options.index(rating) + 1
                if i in [2, 5]:
                    score = 6 - score
                item_scores.append(score)
                st.text(f"{score}分")
        
        inputs["内化同性恋嫌恶得分"] = sum(item_scores)
        
        submitted = st.form_submit_button("提交评估")

    if submitted:
        try:
            # 准备输入数据
            input_data = np.array([
                dict(options_dict["过去一年是否出现自杀行为或想法"])[inputs["过去一年是否出现自杀行为或想法"]],
                dict(options_dict["目前的睡眠情况"])[inputs["目前的睡眠情况"]],
                dict(options_dict["过去一年是否存在颜值焦虑"])[inputs["过去一年是否存在颜值焦虑"]],
                inputs["社会支持得分"],
                inputs["内化同性恋嫌恶得分"]
            ]).reshape(1, -1)

            # 预测结果和概率
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)

            # 根据概率进行分类
            prob_depression = probability[0][1]  # 有抑郁风险的概率

            # 显示结果
            st.subheader("评估结果")
            cols = st.columns(3)
            cols[0].metric("社会支持总分", f"{inputs['社会支持得分']}分")
            cols[1].metric("内化同性恋嫌恶总分", f"{inputs['内化同性恋嫌恶得分']}分")
            cols[2].metric(
                "抑郁风险预测", 
                "有风险" if prediction[0] == 1 else "无风险",
                f"{probability[0][1]*100:.1f}%"
            )
            

            # 特征重要性可视化
            st.subheader("特征重要性")
            feature_importances = model.feature_importances_
            importance_df = pd.DataFrame({
                '特征': feature_names_chinese,
                '重要性': feature_importances
            }).sort_values(by='重要性', ascending=False)

            fig_importance = px.bar(
                importance_df,
                x='重要性',
                y='特征',
                orientation='h',
                color='重要性',
                color_continuous_scale='Blues',
                title='各特征对抑郁风险预测的影响程度'
            )
            st.plotly_chart(fig_importance, use_container_width=True)

            # 用户输入数据可视化
            st.subheader("您的输入数据")
            user_input_df = pd.DataFrame({
                '特征': feature_names_chinese,
                '值': [
                    inputs["过去一年是否出现自杀行为或想法"],
                    inputs["目前的睡眠情况"],
                    inputs["过去一年是否存在颜值焦虑"],
                    str(inputs["社会支持得分"]) + "分",
                    str(inputs["内化同性恋嫌恶得分"]) + "分"
                ]
            })

            fig_user_input = px.bar(
                user_input_df,
                x='值',
                y='特征',
                orientation='h',
                color='值',
                color_continuous_scale='Oranges',
                title='您输入的特征值'
            )
            st.plotly_chart(fig_user_input, use_container_width=True)

        except Exception as e:
            st.error(f"发生错误: {str(e)}")

if __name__ == "__main__":
    main()
