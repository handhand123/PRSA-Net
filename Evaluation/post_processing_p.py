import numpy as np
import pandas as pd
import json
from joblib import Parallel, delayed
import os

def load_json(file):
    with open(file) as f:
        data = json.load(f)
        return data


def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor


def NMS(df, threshold):
    df = df.sort_values(by="score", ascending=False)
    start = np.array(df.xmin.values[:])
    end = np.array(df.xmax.values[:])
    duration = np.array(df.xmax.values[:] - df.xmin.values[:])
    scores = np.array(df.score.values[:])
    index = np.arange(0, len(scores))
    keep = []
    while len(index) > 0:
        p = index[0]
        keep.append(p)
        tt1 = np.maximum(start[p], start[index[1:]])
        tt2 = np.minimum(end[p], end[index[1:]])
        intersection = tt2 - tt1
        Iou = intersection / (duration[p] + duration[index[1:]] - intersection).astype(float)
        inds = np.where(Iou <= threshold)[0]
        index = index[inds + 1]

    new_start = list(df.xmin.values[keep])
    new_end = list(df.xmax.values[keep])
    new_scores = list(df.score.values[keep])
    new_df = pd.DataFrame()
    new_df['xmin'] = new_start
    new_df['xmax'] = new_end
    new_df['score'] = new_scores
    return new_df


def _gen_proposal_video(args, video_name, result):
    files = [os.path.join(args.output['output_path'], f) for f in os.listdir(args.output['output_path']) if video_name in f]

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

    fps = result[video_name]['fps']
    num_frames = result[video_name]['num_frames']
    proposal_list = []
    for j in range(min(args.eval['num_props'], len(df))):
        tmp_proposal = {}
        tmp_proposal["score"] = df.score.values[j]
        tmp_proposal["segment"] = [float(round(max(0, df.xmin.values[j]) / fps, 1)),
                                   float(round(min(num_frames, df.xmax.values[j]) / fps, 1))]
        proposal_list.append(tmp_proposal)
    return {video_name: proposal_list}


def gen_proposal_muticore(args):
    thumos_test_anno = pd.read_csv(args.eval['thumo_test_anno'])

    video_list = thumos_test_anno.video.unique()

    thumos_gt = pd.read_csv(args.eval['thumos_test_gt'])
    result = {
        video:
            {
                'fps': thumos_gt.loc[thumos_gt['video-name'] == video]['frame-rate'].values[0],
                'num_frames': thumos_gt.loc[thumos_gt['video-name'] == video]['video-frames'].values[0]
            }
        for video in video_list
    }
    parallel = Parallel(n_jobs=8)
    generation = parallel(delayed(_gen_proposal_video)(args, video_name, result)
                          for video_name in video_list)
    generation_dict = {}
    [generation_dict.update(d) for d in generation]
    output_dict = {"version": "THUMOS14", "results": generation_dict, "external_data": {}}
    with open(os.path.join(args.output["output_path"], 'generated_result.json'), "w") as out:
        json.dump(output_dict, out)
    print("end")

