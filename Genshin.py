import json
import os
import librosa
import soundfile as sf
from text import cleaners
import os

def make_json(speaker):  # 提取目标人物json
    # 获取原始json数据
    f = open('./result.json', 'r', encoding='utf-8')
    js = f.read()
    f.close()
    # 格式化
    json_all = json.loads("%s" % js)
    sd = {}
    for i in json_all:
        try:
            if "Chinese" in json_all[i]["fileName"]:
                if json_all[i]["npcName"] == speaker and json_all[i]["language"] == 'CHS':
                    sd[i] = {'text': json_all[i]["text"], 'file': json_all[i]["fileName"]}
        except Exception as e:
            print(f'在处理{i}时出现异常=>{e}')
    d = json.dumps(sd, sort_keys=False, indent=4, separators=(',', ':'), ensure_ascii=False)
    f = open('./' + speaker + '.json', 'w', encoding='utf-8')
    f.write(d)
    f.close()


def move_file(pid,speaker):
    os.mkdir('./wavs/p'+str(pid))
    f = open(f'./{speaker}.json', 'r', encoding='utf-8')
    js = f.read()
    f.close()
    all = json.loads("%s" % js)
    k = 0
    f = 0
    j = len(all)
    for i in all:
        try:
            filepath = all[i]['file'].replace('\\','/')
            filepath = filepath.replace('wem','wav')
            src_sig, sr = sf.read(filepath)  # name是要 输入的wav 返回 src_sig:音频数据  sr:原采样频率
            dst_sig = librosa.resample(src_sig, sr, 22050)  # resample 入参三个 音频数据 原采样频率 和目标采样频率
            sf.write(f'./wavs/p{pid}/{i}.wav', dst_sig, 22050)  # 写出数据  参数三个 ：  目标地址  更改后的音频数据  目标采样数据
            k += 1
            print(f'已完成：{k},总数：{j},失败:{f}')
        except Exception as e:
            print(f'处理{i}---{filepath}出现异常。error=>{e}')
            f += 1

def make_filelist(pid,speaker):
    f = open(f'./{speaker}.json', 'r', encoding='utf-8')
    js = f.read()
    f.close()
    all = json.loads("%s" % js)
    fl = open('./filelists/Genshin.txt', 'r', encoding='utf-8').read()
    for i in all:
        if os.path.exists('wavs/p'+str(pid) +'/'+ i + '.wav'):
            try:
                fl = fl + 'wavs/p'+str(pid)+'/' + i + '.wav|' + str(pid) + '|' + cleaners.chinese_cleaners2(all[i]['text']) + '\n'
            except Exception as e:
                print(f'处理{i}({all[i]["text"]})出异常=>{e}')
    ff = open('./filelists/Genshin.txt', 'w', encoding='utf-8')
    ff.write(fl)
    ff.close()

if __name__ == '__main__':
    person_list = ["琴", "安柏", "丽莎", "凯亚", "芭芭拉", "迪卢克", "雷泽", "温迪", "可莉", "班尼特", "诺艾尔", "菲谢尔",
    "砂糖", "莫娜", "迪奥娜", "阿贝多", "优菈", "魈", "北斗", "凝光", "香菱", "行秋", "重云",
    "七七", "刻晴", "达达利亚", "钟离", "辛焱", "甘雨", "胡桃", "烟绯", "申鹤", "云堇", "夜兰", "神里绫华","纳西妲","雷电将军","八重神子","珊瑚宫心海","荒泷一斗"]
    for pid,i in enumerate(person_list):
        make_json(i)
        move_file(pid,i)
        make_filelist(pid,i)

    from sklearn.model_selection import train_test_split

    with open("Genshin.txt", 'r', encoding='utf-8') as f:
        data_list = f.read().split('\n')
        train_datas, test_datas = train_test_split(data_list, train_size=0.8, shuffle=True)

    with open("Genshin-Train.txt", 'w', encoding='utf-8') as f1:
        f1.write("\n".join(train_datas))

    with open("Genshin-Test.txt", 'w', encoding='utf-8') as f2:
        f2.write("\n".join(test_datas))