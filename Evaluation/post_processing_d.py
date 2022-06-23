import numpy as np
import pandas as pd
import json
from joblib import Parallel, delayed
import os

thumos_class = {
    7 : 'BaseballPitch',
    9 : 'BasketballDunk',
    12: 'Billiards',
    21: 'CleanAndJerk',
    22: 'CliffDiving',
    23: 'CricketBowling',
    24: 'CricketShot',
    26: 'Diving',
    31: 'FrisbeeCatch',
    33: 'GolfSwing',
    36: 'HammerThrow',
    40: 'HighJump',
    45: 'JavelinThrow',
    51: 'LongJump',
    68: 'PoleVault',
    79: 'Shotput',
    85: 'SoccerPenalty',
    92: 'TennisSwing',
    93: 'ThrowDiscus',
    97: 'VolleyballSpiking',
}

def load_json(file):
    with open(file) as f:
        data=json.load(f)
        return data

def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor

def NMS(df,threshold):
    df=df.sort_values(by="score", ascending=False)
    start=np.array(df.xmin.values[:])
    end=np.array(df.xmax.values[:])
    duration=np.array(df.xmax.values[:]-df.xmin.values[:])
    scores=np.array(df.score.values[:])
    index=np.arange(0,len(scores))
    keep=[]
    while len(index)>0:
        p=index[0]
        keep.append(p)
        tt1=np.maximum(start[p],start[index[1:]])
        tt2=np.minimum(end[p],end[index[1:]])
        intersection=tt2-tt1
        Iou=intersection/(duration[p]+duration[index[1:]]-intersection).astype(float)
        inds=np.where(Iou<=threshold)[0]
        index=index[inds+1]

    new_start=list(df.xmin.values[keep])
    new_end=list(df.xmax.values[keep])
    new_scores=list(df.score.values[keep])
    new_df=pd.DataFrame()
    new_df['xmin']=new_start
    new_df['xmax']=new_end
    new_df['score']=new_scores
    return new_df

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def _gen_proposal_video(args, video_name, video_cls, thu_label_id, result, topk=2):
    files = [os.path.join(args.output['output_path'], f) for f in os.listdir(args.output['output_path']) if
             video_name in f]

    if len(files) == 0:
        raise ValueError

    dfs = []
    for snippets_file in files:
        snippet_df = pd.read_csv(snippets_file)
        snippet_df = NMS(snippet_df, args.eval['NMS_threshold'])
        dfs.append(snippet_df)
    df = pd.concat(dfs)
    if len(df) > 1:
        df = NMS(df, args.eval['NMS_threshold'])
    df = df.sort_values(by="score", ascending=False)

    video_cls1 = softmax(video_cls)
    video_cls_rank = sorted((e, i) for i, e in enumerate(video_cls1))
    unet_classes = [thu_label_id[video_cls_rank[-k - 1][1]] + 1 for k in range(topk)]
    unet_scores = [video_cls_rank[-k - 1][0] for k in range(topk)]

    fps = result[video_name]['fps']
    num_frames = result[video_name]['num_frames']
    proposal_list = []
    for j in range(min(args.eval['num_props'], len(df))):
        for k in range(topk):
            tmp_proposal = {}
            tmp_proposal['label'] = thumos_class[int(unet_classes[k])]
            tmp_proposal["score"] = float(df.score.values[j] * unet_scores[k])
            tmp_proposal["segment"] = [float(round(max(0, df.xmin.values[j]) / fps, 4)),
                                       float(round(min(num_frames, df.xmax.values[j]) / fps, 4))]
            proposal_list.append(tmp_proposal)
    index = np.argsort([t["score"] for t in proposal_list])
    proposal_list = [proposal_list[idx] for idx in index[::-1]]
    return {video_name: proposal_list}


def gen_proposal_muticore(args):

    thumos_test_anno = pd.read_csv(args.eval['thumo_test_anno'])
    video_list = thumos_test_anno.video.unique()
    thu_label_id = np.sort(thumos_test_anno.type_idx.unique())[1:] - 1
    thu_video_id = np.array([int(i[-4:]) - 1 for i in video_list])
    cls_data = np.load("./data/uNet_test.npy")
    cls_data = cls_data[thu_video_id, :][:, thu_label_id]
    thumos_gt = pd.read_csv(args.eval['thumos_test_gt'])
    result = {
        video:
            {
                'fps': thumos_gt.loc[thumos_gt['video-name'] == video]['frame-rate'].values[0],
                'num_frames': thumos_gt.loc[thumos_gt['video-name'] == video]['video-frames'].values[0]
            }
        for video in video_list
    }
    parallel = Parallel(n_jobs=16)
    generation = parallel(delayed(_gen_proposal_video)(args, video_name, video_cls, thu_label_id, result)
                          for video_name, video_cls in zip(video_list, cls_data))
    generation_dict = {}
    [generation_dict.update(d) for d in generation]
    output_dict = {"version": "THUMOS14", "results": generation_dict, "external_data": {}}
    with open(os.path.join(args.output["output_path"], 'generated_result.json'), "w") as out:
        json.dump(output_dict, out)
    print("end")