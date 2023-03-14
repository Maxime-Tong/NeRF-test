# nohup python -u run_with_eval.py --config configs/lego.txt > ./log/lego.log 2>&1 &
nohup python -u run_with_eval.py --config configs/lego.txt --expname blender_paper_lego_diffpsnr> ./log/lego_dif_psnrs.log 2>&1 &
