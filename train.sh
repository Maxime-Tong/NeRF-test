# nohup python -u run_with_eval.py --config configs/lego.txt > ./log/lego.log 2>&1 &
# nohup python -u run_with_eval.py --config configs/lego.txt --expname blender_paper_lego_inc_olc_psnr > ./log/lego_inc_olc_psnr.log 2>&1 &
nohup python -u run_with_entropy.py --config configs/lego.txt --expname blender_paper_lego_alpha_entropy_10_10 --i_print 10 > ./log/lego_alpha_entropy_10_10.log 2>&1 &