import numpy as np
import matplotlib.pyplot as plt
import os

def equirectangular_great_circle(longitude_array, azimuth, inclination):
	"""
	U(原点)^のカーブが1周期(原点→上がる→原点+180度→下がる→原点)
	longitude_array [degree] : array of longitude(経度) samples
	azimuth(方位角) [rad]
	inclination(傾斜) [rad] : 縦方向のカーブの最大値(角度)。azimuth+90度でなる。
	"""
	return np.rad2deg(np.arctan(np.tan(inclination) * (np.sin(np.deg2rad(longitude_array) - azimuth))))

def set_equirectangluar_plot():
	plt.xlim(-180, 180)
	plt.ylim(-90, 90)
	plt.xticks(np.arange(-180.0,181.0, step=45.0))
	plt.yticks(np.arange(-90.0,91.0, step=30.0))
	plt.xlabel('Longitude [degree]')
	plt.ylabel('Lattitude [degree]')
	plt.grid()
	return

# 手首と肩の角度(横,縦)をもらって大円を書くためのパラメータを求める
def cal_alpha_beta(sholder:list, wrist:list): # 手首と肩の順番は関係ない
	sholder = [np.deg2rad(n) for n in sholder]
	wrist = [np.deg2rad(n) for n in wrist]
	# print(f'肩{sholder},手首{wrist}ラジアン')
	A = np.array([[np.cos(sholder[0]), np.sin(sholder[0])], 
	  [np.cos(wrist[0]), np.sin(wrist[0])]])
	B = np.array([np.tan(sholder[1]), np.tan(wrist[1])])
	# x = np.linalg.solve(A, B)
	# print(f'solve x={x}')
	x = np.linalg.lstsq(A, B, rcond=None)[0] # 手首と肩が同じ座標のときも解を求める
	# print(f'A={A},B={B},x={x}')
	beta = np.arctan(np.sqrt(x[0]*x[0] + x[1]*x[1]))
	alpha = np.arcsin(-x[0]/np.tan(beta))
	if x[1] / np.tan(beta) < 0:
		alpha = np.pi - alpha
	# print(f'alpha{alpha},beta{beta}')
	return alpha, beta # ラジアン表記

# main関数
# def get(sholder=[297,7], wrist=[312,21], file='inputOmni/R0010162.JPG', output='greatOmni/', save=True):
def get(sholder=[30,20], wrist=[30,-40], file='inputOmni/R0010148.JPG', output='greatOmni/', save=True):
	N_samples = 1080 # 要素数0~1079
	longitude_samples = np.rad2deg(np.linspace(-np.pi, np.pi, N_samples)) # 横軸の細かさ
	gc = []
	if sholder[1] == 0 and wrist[1] == 0:
		gc = np.zeros(N_samples) # 赤道
	elif sholder[0] == wrist[0]: # 経線になるはず
		# 右方向に進むとする
		if sholder[1] > wrist[1]: # 床を指差してる
			inclination = - 60 # fov60度だからー90度じゃなくても床まで行く
		else:
			inclination = 60 # +fov/2=90
		x = np.linspace(0, 1, N_samples) # longitude_samplesを使うとうまく書けなかった
		gc = inclination * np.sin((2*np.pi*x - np.deg2rad(sholder[0]))*6) # 周期6倍,経線の座標が原点
	else:
		azimuth, inclination = cal_alpha_beta(sholder, wrist)
		gc = equirectangular_great_circle(longitude_samples, azimuth, inclination)

	if save:
		plt.figure()
		img = plt.imread(file)
		plt.imshow(img, zorder=0, extent=[-180, 180, -90, 90], alpha=0.75)
		plt.plot(longitude_samples, gc, color='red')
		set_equirectangluar_plot()
		output = os.path.join(output, os.path.splitext(os.path.basename(file))[0] + '.png')
		plt.savefig(output, dpi=200)
		# plt.show()
	return longitude_samples, gc # 横軸のデータ, 縦軸のデータ

if __name__ == '__main__':
	get()