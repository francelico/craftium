import os
import shutil
import time
import pandas as pd
from tqdm import tqdm
import utils3d
from dataclasses import dataclass
import tyro
from util import get_file_hash

def collect_hash_identifiers(output_dir):
    paths = []
    rootdir = os.path.join(output_dir, 'raw')
    for envdataset_dir in os.listdir(rootdir):
        for instance_dir in os.listdir(os.path.join(rootdir, envdataset_dir)):
            paths.append(os.path.join('raw', envdataset_dir, instance_dir))

    with ThreadPoolExecutor(max_workers=max(os.cpu_count(), len(paths))) as executor, \
            tqdm(total=len(paths), desc="Extracting") as pbar:
        def worker(path: str) -> str:
            try:
                rawdata_path = os.path.join(output_dir, path, 'data.npz')
                assert os.path.exists(rawdata_path), f"Data file not found for {path}"
                sha256 = get_file_hash(rawdata_path)
                pbar.update()
                return sha256
            except Exception as e:
                pbar.update()
                print(f"Error extracting for {path}: {e}")
                return None


        sha256s = executor.map(worker, paths)
        executor.shutdown(wait=True)
    return pd.DataFrame(zip(sha256s, paths), columns=['sha256', 'local_path'])

def need_process(key):
    return key in args.field or args.field == ['all']

@dataclass
class Args: # Args(DatasetArgs) is also possible if we want Dataset specific arguments in the future
    """Command line arguments for the program."""
    output_dir: str
    """Directory to save the metadata"""
    field: str = "all"
    """Fields to process, separated by commas"""
    from_file: bool = False
    """Build metadata from file instead of from records of processings.
    Useful when some processing fail to generate records but file already exists."""
    debug: bool = False
    """Enable debug mode. Will run the code in a single process."""

if __name__ == '__main__':
    args = tyro.cli(Args)
    if args.debug:
        from util import MockThreadPoolExecutor as ThreadPoolExecutor
    else:
        from concurrent.futures import ThreadPoolExecutor

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'merged_records'), exist_ok=True)

    args.field = args.field.split(',')

    timestamp = str(int(time.time()))

    # get file list
    if os.path.exists(os.path.join(args.output_dir, 'metadata.csv')):
        print('Loading previous metadata...')
        metadata = pd.read_csv(os.path.join(args.output_dir, 'metadata.csv'))
    else:
        metadata = collect_hash_identifiers(args.output_dir)
    metadata.set_index('sha256', inplace=True)

    # merge generated
    df_files = [f for f in os.listdir(args.output_dir) if f.startswith('downloaded_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(args.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index('sha256', inplace=True)
        if 'local_path' in metadata.columns:
            metadata.update(df, overwrite=True)
        else:
            metadata = metadata.join(df, on='sha256', how='left')
        for f in df_files:
            shutil.move(os.path.join(args.output_dir, f),
                        os.path.join(args.output_dir, 'merged_records', f'{timestamp}_{f}'))

    # detect models
    image_models = []
    if os.path.exists(os.path.join(args.output_dir, 'features')):
        image_models = os.listdir(os.path.join(args.output_dir, 'features'))
    latent_models = []
    if os.path.exists(os.path.join(args.output_dir, 'latents')):
        latent_models = os.listdir(os.path.join(args.output_dir, 'latents'))
    ss_latent_models = []
    if os.path.exists(os.path.join(args.output_dir, 'ss_latents')):
        ss_latent_models = os.listdir(os.path.join(args.output_dir, 'ss_latents'))
    print(f'Image models: {image_models}')
    print(f'Latent models: {latent_models}')
    print(f'Sparse Structure latent models: {ss_latent_models}')
    if 'voxelized' not in metadata.columns:
        metadata['voxelized'] = [False] * len(metadata)
    if 'num_voxels' not in metadata.columns:
        metadata['num_voxels'] = [0] * len(metadata)
    if 'cond_rendered' not in metadata.columns:
        metadata['cond_rendered'] = [False] * len(metadata)
    for model in image_models:
        if f'feature_{model}' not in metadata.columns:
            metadata[f'feature_{model}'] = [False] * len(metadata)
    for model in latent_models:
        if f'latent_{model}' not in metadata.columns:
            metadata[f'latent_{model}'] = [False] * len(metadata)
    for model in ss_latent_models:
        if f'ss_latent_{model}' not in metadata.columns:
            metadata[f'ss_latent_{model}'] = [False] * len(metadata)

    # merge voxelized
    df_files = [f for f in os.listdir(args.output_dir) if f.startswith('voxelized_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(args.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index('sha256', inplace=True)
        metadata.update(df, overwrite=True)
        for f in df_files:
            shutil.move(os.path.join(args.output_dir, f),
                        os.path.join(args.output_dir, 'merged_records', f'{timestamp}_{f}'))

    # merge cond_rendered
    df_files = [f for f in os.listdir(args.output_dir) if f.startswith('cond_rendered_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(args.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index('sha256', inplace=True)
        metadata.update(df, overwrite=True)
        for f in df_files:
            shutil.move(os.path.join(args.output_dir, f),
                        os.path.join(args.output_dir, 'merged_records', f'{timestamp}_{f}'))

    # merge features
    for model in image_models:
        df_files = [f for f in os.listdir(args.output_dir) if f.startswith(f'feature_{model}_') and f.endswith('.csv')]
        df_parts = []
        for f in df_files:
            try:
                df_parts.append(pd.read_csv(os.path.join(args.output_dir, f)))
            except:
                pass
        if len(df_parts) > 0:
            df = pd.concat(df_parts)
            df.set_index('sha256', inplace=True)
            metadata.update(df, overwrite=True)
            for f in df_files:
                shutil.move(os.path.join(args.output_dir, f),
                            os.path.join(args.output_dir, 'merged_records', f'{timestamp}_{f}'))

    # merge latents
    for model in latent_models:
        df_files = [f for f in os.listdir(args.output_dir) if f.startswith(f'latent_{model}_') and f.endswith('.csv')]
        df_parts = []
        for f in df_files:
            try:
                df_parts.append(pd.read_csv(os.path.join(args.output_dir, f)))
            except:
                pass
        if len(df_parts) > 0:
            df = pd.concat(df_parts)
            df.set_index('sha256', inplace=True)
            metadata.update(df, overwrite=True)
            for f in df_files:
                shutil.move(os.path.join(args.output_dir, f),
                            os.path.join(args.output_dir, 'merged_records', f'{timestamp}_{f}'))

    # merge sparse structure latents
    for model in ss_latent_models:
        df_files = [f for f in os.listdir(args.output_dir) if f.startswith(f'ss_latent_{model}_') and f.endswith('.csv')]
        df_parts = []
        for f in df_files:
            try:
                df_parts.append(pd.read_csv(os.path.join(args.output_dir, f)))
            except:
                pass
        if len(df_parts) > 0:
            df = pd.concat(df_parts)
            df.set_index('sha256', inplace=True)
            metadata.update(df, overwrite=True)
            for f in df_files:
                shutil.move(os.path.join(args.output_dir, f),
                            os.path.join(args.output_dir, 'merged_records', f'{timestamp}_{f}'))

    # build metadata from files
    if args.from_file:
        with (ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
              tqdm(total=len(metadata), desc="Building metadata") as pbar):
            def worker(sha256):
                try:
                    if need_process('voxelized'):
                        try:
                            assert os.path.exists(os.path.join(args.output_dir, 'voxels', f'{sha256}.ply'))
                            pts = utils3d.io.read_ply(os.path.join(args.output_dir, 'voxels', f'{sha256}.ply'))[0]
                            metadata.loc[sha256, 'voxelized'] = True
                            metadata.loc[sha256, 'num_voxels'] = len(pts)
                        except Exception as e:
                            metadata.loc[sha256, 'voxelized'] = False
                            metadata.loc[sha256, 'num_voxels'] = 0
                    if need_process('cond_rendered'):
                        metadata.loc[sha256, 'cond_rendered'] = True \
                            if os.path.exists(os.path.join(args.output_dir, 'renders_cond', sha256, 'transforms.json')) \
                            else False
                    for model in image_models:
                        if need_process(f'feature_{model}'):
                            metadata.loc[sha256, f'feature_{model}'] = True \
                                if os.path.exists(os.path.join(args.output_dir, 'features', model, f'{sha256}.npz')) \
                                else False
                    for model in latent_models:
                        if need_process(f'latent_{model}'):
                            metadata.loc[sha256, f'latent_{model}'] = True \
                                if os.path.exists(os.path.join(args.output_dir, 'latents', model, f'{sha256}.npz')) \
                                else False
                    for model in ss_latent_models:
                        if need_process(f'ss_latent_{model}'):
                            metadata.loc[sha256, f'ss_latent_{model}'] = True \
                                if os.path.exists(os.path.join(args.output_dir, 'ss_latents', model, f'{sha256}.npz')) \
                                else False
                    pbar.update()
                except Exception as e:
                    print(f'Error processing {sha256}: {e}')
                    pbar.update()


            executor.map(worker, metadata.index)
            executor.shutdown(wait=True)

    # statistics
    metadata.to_csv(os.path.join(args.output_dir, 'metadata.csv'))
    num_generated = metadata['local_path'].count() if 'local_path' in metadata.columns else 0
    with open(os.path.join(args.output_dir, 'statistics.txt'), 'w') as f:
        f.write('Statistics:\n')
        f.write(f'  - Number of assets: {len(metadata)}\n')
        f.write(f'  - Number of assets generated: {num_generated}\n')
        f.write(f'  - Number of assets voxelized: {metadata["voxelized"].sum()}\n')
        if len(image_models) != 0:
            f.write(f'  - Number of assets with image features extracted:\n')
            for model in image_models:
                f.write(f'    - {model}: {metadata[f"feature_{model}"].sum()}\n')
        if len(latent_models) != 0:
            f.write(f'  - Number of assets with latents extracted:\n')
            for model in latent_models:
                f.write(f'    - {model}: {metadata[f"latent_{model}"].sum()}\n')
        if len(ss_latent_models) != 0:
            f.write(f'  - Number of assets with sparse structure latents extracted:\n')
            for model in ss_latent_models:
                f.write(f'    - {model}: {metadata[f"ss_latent_{model}"].sum()}\n')
        f.write(f'  - Number of assets with image conditions: {metadata["cond_rendered"].sum()}\n')

    with open(os.path.join(args.output_dir, 'statistics.txt'), 'r') as f:
        print(f.read())