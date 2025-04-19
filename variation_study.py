import os
import pandas as pd

win_lens = [125, 250, 500, 1000]
thresholds = [0.10, 0.20, 0.30, 0.4]

result_table = []

for win_len in win_lens:
    win_shift = int(win_len / 4)
    
    for threshold in thresholds:
        # Run the command
        cmd = f"""
        python process.py --audio_dir custom_sound \
            --output_path output_sound \
            --time_clip 1 \
            --target_fs_values 16000 \
            --clipping_thresholds {threshold} \
            --dynamic 1 \
            --saving 0 \
            --plotting 0 \
            --delta 300 \
            --win_len {win_len} \
            --win_shift {win_shift}
        """
        os.system(cmd)

        # Load the Excel file
        folder = f"output_sound/fs_16000_threshold_{threshold:.2f}"
        file_path = os.path.join(folder, "results_1s.xlsx")
        
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)

            # Extract the relevant values
            pesq_mean = df['pesq_imp_mean_array'].values[0]
            pesq_std = df['pesq_imp_std_array'].values[0]
            sdr_mean = df['sdr_imp_mean_array'].values[0]
            sdr_std = df['sdr_imp_std_array'].values[0]

            result_table.append({
                'win_len': win_len,
                'threshold': threshold,
                'pesq_mean': pesq_mean,
                'pesq_std': pesq_std,
                'sdr_mean' : sdr_mean,
                'sdr_std' : sdr_std
            })
        else:
            print(f"Missing file: {file_path}")

# Save the summary
summary_df = pd.DataFrame(result_table)
summary_df.to_excel("variation_study_summary.xlsx", index=False)
print("Saved summary to variation_study_summary.xlsx")
