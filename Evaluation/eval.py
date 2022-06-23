import numpy as np
import matplotlib.pyplot as plt
from Evaluation.eval_proposal import ANETproposal
from Evaluation.eval_detection import ANETdetection


def run_evaluation(ground_truth_file, proposal_file, dataset_name='',
                   max_avg_nr_proposal=100,
                   tiou_thre=np.linspace(0.5, 1.0, 11), subset='test'):
    anet_proposal = ANETproposal(ground_truth_file, proposal_file,
                                 dataset_name=dataset_name,
                                 tiou_thresholds=tiou_thre, max_avg_nr_proposals=max_avg_nr_proposal,
                                 subset=subset, verbose=True, check_status=False)

    anet_proposal.evaluate()

    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposal = anet_proposal.proposals_per_video

    return (average_nr_proposal, average_recall, recall)


def plot_metric(args, average_nr_proposals, average_recall, recall, tiou_thresholds=np.linspace(0.5, 1.0, 11)):
    fn_size = 14
    plt.figure(num=None, figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    colors = ['k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo']
    area_under_curve = np.zeros_like(tiou_thresholds)
    for i in range(recall.shape[0]):
        area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)

    for idx, tiou in enumerate(tiou_thresholds[::2]):
        ax.plot(average_nr_proposals, recall[2 * idx, :], color=colors[idx + 1],
                label="tiou=[" + str(tiou) + "],area=" + str(int(area_under_curve[2 * idx] * 100) / 100.),
                linewidth=4, linestyle='-', marker=None)

    ax.plot(average_nr_proposals, average_recall, color=colors[0],
            label="tiou=0.5:0.1:1.0," + "area=" + str(int(np.trapz(average_recall, average_nr_proposals) * 100) / 100.),
            linewidth=4, linestyle='-', marker=None)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1], loc='best')

    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
    plt.savefig(args.eval["save_fig_path"])


def evaluation_proposal(args):
    uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid = run_evaluation(
        args.eval['thumos_gt'],
        args.output["output_path"] + '/generated_result.json',
        args.dataset['dataset_name'],
        max_avg_nr_proposal=1000,
        tiou_thre=np.linspace(0.5, 1.0, 11),
        subset='test')

    plot_metric(args, uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid)
    print("AR@1 is \t", np.mean(uniform_recall_valid[:, 0]))
    print("AR@5 is \t", np.mean(uniform_recall_valid[:, 4]))
    print("AR@10 is \t", np.mean(uniform_recall_valid[:, 9]))
    print("AR@50 is \t", np.mean(uniform_recall_valid[:, 49]))
    print("AR@100 is \t", np.mean(uniform_recall_valid[:, 99]))
    print("AR@200 is \t", np.mean(uniform_recall_valid[:, 199]))
    print("AR@500 is \t", np.mean(uniform_recall_valid[:, 499]))
    print("AR@1000 is \t", np.mean(uniform_recall_valid[:, -1]))


def evalution_detection(args):
    ground_truth_filename = "./Evaluation/thumos_gt.json"
    tious = [0.3, 0.4, 0.5, 0.6, 0.7]
    anet_detection = ANETdetection(ground_truth_filename, args.output["output_path"] + '/generated_result.json',
                                   dataset_name=args.dataset['dataset_name'],
                                   tiou_thresholds=tious,
                                   subset='test', verbose=True, check_status=False)
    mAPs, average_mAP = anet_detection.evaluate()
    for (tiou, mAP) in zip(tious, mAPs):
        print("mAP at tIoU {} is {}".format(tiou, mAP))

