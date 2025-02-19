rm -rf ckpt
echo -e "Downloading Motion-Agent ckpts"
gdown --fuzzy https://drive.google.com/file/d/1Tagt2xUwv_h0JNMtrM_Ty1rWemkLF5jH/view

unzip motion_agent.zip

echo -e "Cleaning\n"
rm motion_agent.zip
echo -e "Downloading done!"