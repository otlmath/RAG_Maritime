import google.generativeai as genai
import glob
import os

try:
    os.mkdir(r'.\TXT_DATA2')
except:
    pass 

with open("gemini_api_key.txt", "r") as f:
	api_key = f.readline().strip()
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash-lite")

def chunking(lines):
    chunks=[]
    chunk=''
    for line in lines:
        if len(chunk)<8000:
            chunk= chunk+line
        else:
            if line.strip().split(' ')[0].split('.')[0].isdigit():
                chunks.append(chunk)
                chunk=line
            else:
                chunk=chunk+line
    chunks.append(chunk)
    return(chunks)

prompt='The following text was read by an OCR and therefore has incorrect spacings and line breaks. Please fix this so that line breaks happen by paragraphs, not in the middle of sentence or words. Be aware that contents are itemized using digits and dots. Reply ONLY with the cleaned text. Begin text:\n'


text_data=glob.glob(r'.\TXT_DATA\*.txt')

for fpath in text_data: 
    print("Working on",fpath)
    with open(fpath, 'r',encoding='utf8') as file:
        lines = file.readlines()
    chunks= chunking(lines)
    print(len(chunks),'chunks')
    cleaned=[]

    for chunk in chunks:
        response = model.generate_content(prompt+chunk)
        if response.text:
            if len(response.text)>20:
                print(response.text[:20])
            cleaned.append(response.text)
        else:
            print('\n\nNO RESPONSE!!!!!\n\n',chunk)
    outpath = fpath.replace('TXT_DATA','TXT_DATA2').replace('.txt','_cleaned.txt')
    with open(outpath, 'w',encoding='utf8') as file:
        file.writelines(cleaned)
        