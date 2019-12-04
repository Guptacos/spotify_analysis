# Install and update pip
sudo apt update && sudo apt install python3-pip
y

sudo python3 -m pip uninstall pip && sudo apt install python3-pip --reinstall

pip3 install -r requirements.txt && python3 getTopSongs.py
