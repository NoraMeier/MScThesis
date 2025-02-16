import matplotlib.pyplot as plt
import seaborn
import argparse
import os
import cv2
import numpy as np
import tqdm

MAX_UNCERTAINTY = 3.0
MAX_ERROR = 2.3
FILTER_PERCENTILE = 97
UNCERTAINTY_TYPES = ['ensemble', 'featextr', 'flowest', 'refine']

def make_density_plot(std_values, error_values, correlation, video, uncertainty_type, idx, filtered: bool, no_unc, max_unc=MAX_UNCERTAINTY, max_err=MAX_ERROR):
    seaborn.set_theme(style='ticks')
    ax = seaborn.jointplot(x=std_values, y=error_values, kind='hex', color='#4CB391')
    ax.ax_marg_x.set_xlim(0.0, max_unc)
    ax.ax_marg_y.set_ylim(0.0, max_err)
    ax.set_axis_labels("Uncertainty", "Error")
    ax.figure.suptitle(f"Density plot{ ' (filtered)' if filtered else ''}. Correlation coefficient: {correlation}")
    plot_type = "densityplot_filtered" if filtered else "densityplot_full"
    ax.figure.savefig(f"video/{video}/{plot_type}_{no_unc}{uncertainty_type}/frame_{idx}.jpg")
    plt.close(ax.figure)

def convert_img_to_array(img: np.ndarray):
    summed_img = img.sum(axis=2)
    flattened_image = summed_img.flatten()
    return flattened_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='catvideo', type=str, choices=['catvideo', 'bg3', 'jojos', 'vimeo'])
    parser.add_argument('--no_unc', default='',type=str, choices=['', 'compare_to_no_unc_'])
    args = parser.parse_args()

    correlations = {}
    for uncertainty_type in UNCERTAINTY_TYPES:
        correlations[uncertainty_type] = []

    max_error = 0.0
    max_uncertainty = 0.0

    for uncertainty_type in UNCERTAINTY_TYPES:
        frame_path = f"video/{args.video}/interpolated_frames_{uncertainty_type}"
        no_unc_frame_path = f"video/{args.video}/interpolated_frames_none"
        uncertainty_frame_files = os.listdir(frame_path)

        denseplot_full_path = f"video/{args.video}/densityplot_full_{args.no_unc}{uncertainty_type}"
        if not os.path.exists(denseplot_full_path):
            os.mkdir(denseplot_full_path)
        denseplot_filtered_path = f"video/{args.video}/densityplot_filtered_{args.no_unc}{uncertainty_type}"
        if not os.path.exists(denseplot_filtered_path):
            os.mkdir(denseplot_filtered_path)

        for idx in tqdm.tqdm(range(len(uncertainty_frame_files))):
            img = cv2.imread(f"{frame_path}/frame_{idx}.jpg")
            img = img.astype(np.float32) / 255.0
            one_img_height, one_img_width = int(img.shape[0] / 2), int(img.shape[1] / 2)
            gt_img = img[:one_img_height, :one_img_width, :]
            pred_img = img[:one_img_height, one_img_width:, :]
            if args.no_unc == "compare_to_no_unc_":
                full_img = cv2.imread(f"{no_unc_frame_path}/frame_{idx}.jpg")
                full_img = full_img.astype(np.float32) / 255.0
                pred_img_width = int(full_img.shape[1] / 2)
                pred_img = full_img[:, pred_img_width:, :]
                gt_img = full_img[:, :pred_img_width, :]
            error_img = abs(gt_img - pred_img)
            error_values = convert_img_to_array(error_img)
            max_error = round(max(max_error, error_values.max()), 2)

            std_img = img[one_img_height:, one_img_width:, :]
            std_values = convert_img_to_array(std_img)
            max_uncertainty = round(max(max_uncertainty, std_values.max()), 2)

            cor = np.corrcoef(std_values, error_values)
            cor = round(cor[0, 1], 2)
            (correlations[uncertainty_type]).append(cor)

            make_density_plot(std_values, error_values, cor, args.video, uncertainty_type, idx, False, no_unc=args.no_unc, max_err=0.5, max_unc=0.5)

            dist_arr = error_values * error_values + std_values * std_values
            threshold = np.percentile(dist_arr, FILTER_PERCENTILE)
            valid_idxs = np.where(dist_arr > threshold)[0]
            filtered_errors = error_values[valid_idxs]
            filtered_stds = std_values[valid_idxs]

            make_density_plot(filtered_stds, filtered_errors, cor, args.video, uncertainty_type, idx, True, no_unc=args.no_unc)

    for key in correlations.keys():
        val = correlations[key]
        avg_cor = np.mean(val)
        print(f"{key} average correlation: {avg_cor}")
    x_vals = np.array(range(len(correlations['featextr'])))

    #y_vals = [(correlations['ensemble'][i], correlations['featextr'][i], correlations['flowest'][i], correlations['refine'][i]) for i in range(len(correlations['featextr']))]
    #y_vals = [np.array(cor_list) for cor_list in correlations.values()]
    #plt.plot(x_vals, np.c_[y_vals[0], y_vals[1], y_vals[2], y_vals[3]])
    print(correlations.keys())
    print(correlations['featextr'])
    print(correlations['ensemble'])
    print(correlations['flowest'])
    for uncertainty_type in UNCERTAINTY_TYPES:
        plt.plot(x_vals, np.array(correlations[uncertainty_type]))

    plt.xlabel("Frames")
    plt.ylabel("Correlation")
    plt.title(f"Correlation between uncertainty and error per frame.")
    plt.ylim((0.0, 1.0))
    plt.legend(UNCERTAINTY_TYPES)
    plt.savefig(f"video/{args.video}/correlation_curves.jpg")

    print(f"Max error: {max_error}, max uncertainty: {max_uncertainty}")


if __name__ == "__main__":
    main()