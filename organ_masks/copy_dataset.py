import pandas as pd
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

#cases += pd.read_csv('/dev/shm/pedro/foundational/data_code/UCSF_ids_1234.csv')["BDMAP_ID"].tolist()

# Directories
source_ct_dir = '/projects/bodymaps/Pedro/data/radiologist_annotations_merlin_ucsf_atlas_multi_cancer/'
source_mask_dir = '/projects/bodymaps/Pedro/data/Dataset315_lesion_only_radiologist_train_Mar_18_ucsf_with_merlin_corrected/'
dest_ct_dir = '/projects/bodymaps/Pedro/data/nnUNet_raw/Dataset315_lesion_only_radiologist_train_Mar_18_ucsf_with_merlin_corrected/imagesTr'#keep imagesTr
dest_mask_dir = '/projects/bodymaps/Pedro/data/nnUNet_raw/Dataset315_lesion_only_radiologist_train_Mar_18_ucsf_with_merlin_corrected/labelsTr'#keep labelsTr

# Load cases from CSV files
cases = [f for f in os.listdir(source_mask_dir) if 'BDMAP' in f]


# Create destination directories if they don't exist
os.makedirs(dest_ct_dir, exist_ok=True)
os.makedirs(dest_mask_dir, exist_ok=True)

def copy_case(case):
    """
    Copies the CT and mask for a given case.
    Returns (case, status, detail).
    """
    source_ct = os.path.join(source_ct_dir, case, 'ct.nii.gz')
    source_mask = os.path.join(source_mask_dir, case, 'combined_labels.nii.gz')
    dest_ct = os.path.join(dest_ct_dir, f'{case}_0000.nii.gz')
    dest_mask = os.path.join(dest_mask_dir, f'{case}.nii.gz')

    if not os.path.exists(source_ct):
        return case, 'ct_not_found', source_ct
    if not os.path.exists(source_mask):
        return case, 'mask_not_found', source_mask

    try:
        shutil.copy(source_ct, dest_ct)
        shutil.copy(source_mask, dest_mask)
        return case, 'copied', None
    except Exception as e:
        return case, 'copy_error', str(e)

def parallel_copy(cases, max_workers=8):
    """
    Copies files for all cases in parallel with a progress bar.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(copy_case, case): case for case in cases}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Copying cases"):
            results.append(future.result())

    # Aggregate
    by_status = {}
    for case, status, detail in results:
        by_status.setdefault(status, []).append((case, detail))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total cases attempted:  {len(results)}")
    print(f"  -> copied:            {len(by_status.get('copied', []))}")
    print(f"  -> ct_not_found:      {len(by_status.get('ct_not_found', []))}")
    print(f"  -> mask_not_found:    {len(by_status.get('mask_not_found', []))}")
    print(f"  -> copy_error:        {len(by_status.get('copy_error', []))}")
    print("=" * 60)

    for status in ['ct_not_found', 'mask_not_found', 'copy_error']:
        items = by_status.get(status, [])
        if items:
            print(f"\n[{status}] {len(items)} case(s). Examples (up to 10):")
            for case, detail in items[:10]:
                print(f"  {case}: {detail}")

# Run the parallel copying
if __name__ == "__main__":
    print(f"Source mask dir: {source_mask_dir}")
    print(f"Source CT dir:   {source_ct_dir}")
    print(f"Cases found in source_mask_dir: {len(cases)}")
    parallel_copy(cases, max_workers=16)