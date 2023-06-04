from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
import pickle
import time

app = Flask(__name__)

# モデルの読み込み
model = joblib.load('./data/model.pkl')

#ラベルエンコーダーの読み込み
with open('./data/label.pkl', mode='rb') as fp:
    le=pickle.load(fp)

#データフレームの読み込み
df=pd.read_csv("./data/all_pokemon_syuzoku&type&top30.csv",index_col=0)
#一部の列の抽出
df=df[['ポケモン名','オリジナルのタイプ1', 'オリジナルのタイプ2', 'HP', '攻撃', '防御', '特攻', '特防', '素早',
       'ノーマル', 'ほのお', 'みず', 'でんき', 'くさ', 'こおり', 'かくとう', 'どく', 'じめん', 'ひこう',
       'エスパー', 'むし', 'いわ', 'ゴースト', 'ドラゴン', 'あく', 'はがね', 'フェアリー','カイリュー','サーフゴー',
 'ハバタクカミ']]

#6匹のデータフレームの作成
def get_inputdf(plist):
    for i in range(len(plist)):
        if i==0:
            input_df=df[df["ポケモン名"]==(plist[i])]
        else:
            input_df=pd.concat([input_df,df[df["ポケモン名"]==(plist[i])]])
    
    #インデックス番号の整理
    input_df=input_df.reset_index(drop=True)
    #パーティ和の作成
    for i in input_df.columns[27:]:
        input_df[i+"パーティ和"]=sum(input_df[i])
    
    #パーティ和から自身の和を引く
    for i in range(len(input_df)):
        for j in range(30,33):
            input_df.loc[i,input_df.columns[j]]=input_df.loc[i,input_df.columns[j]]-input_df.loc[i,input_df.columns[j-3]]
    
    return input_df

def predict_pokemon(input_df,tera_name,flag):
    #予測するポケモンのデータフレームを取得
    machine_input=input_df[input_df["ポケモン名"]==tera_name]
    #結果の確率の辞書
    result_dict={}
    try:
        #それぞれのラベルの予測を保持
        probabilities = model.predict_proba(machine_input.loc[:,machine_input.columns[3:]])
    except ValueError:
        flag=False
        
    return probabilities,flag

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #６匹のデータを取得
    flag=True
    poke_list = []
    for i in range(1, 7):
        list_name = request.form.get('poke_name_' + str(i))
        poke_list.append(list_name)

    #テラスタイプを予測したいポケモン名を取得
    tera_name=request.form.get('tera_name')
    #6匹のデータフレームを関数から取得
    input_df=get_inputdf(poke_list)
    #確率とラベルを保持
    result,flag=predict_pokemon(input_df,tera_name,flag)
    return result
    

if __name__ == '__main__':
    app.run(debug=True)
