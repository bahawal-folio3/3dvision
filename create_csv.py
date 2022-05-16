import os
import pandas as pd 
from tqdm import tqdm
def get_dataframe(classes,datadir,window=8):
    
    data = []
    for i in tqdm(range(len(classes))):
        videos = os.listdir(f'{datadir}/{classes[i]}')
        for video in videos:
            video_path = f"{datadir}/{classes[i]}/{video}/"
            for j in range(len(os.listdir(video_path))-window): # no of frame = windows size so we don't go out of bound
                sliding_window = []
                for k in range(j,window+j): # +1 to include last window
                    sliding_window.append(f'{video_path}{k}.jpg')
                data.append([sliding_window,i])


    df = pd.DataFrame(data)
    df.set_axis(['img','label'],axis=1,inplace=True)
    return df

if __name__ == "__main__":
    classes = ['everythingelse','service']
    df = get_dataframe(classes,'data',window=8)
    print(df.shape)
    print(df.head())
    print(len(df.img[0]))
    print(df.img[0])
    print(df['label'].value_counts())