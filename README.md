## 链接
	https://github.com/ouprince/textcnn-cancha.git
	
## 说明
	1.解决了大数据量运行常见的内存溢出问题
	2.加深了 textcnn 网络深度， 并且引入 残差结构 强化深度模型学习能力
	3.默认 gpu 运行
	
## 使用方法
	1.将训练数据和验证数据放在 data 目录 数据格式: label + \t + title + \t + content
	2.假设训练数据为 train.data  验证数据 dev.data
	3.修改 dataset.py 里面第 19 行， train = 'train.data' , validation = 'dev.data'
	4.调整 main.py 中相应的参数后， 直接运行 python main.py 进行训练即可
	5.训练完成后， predict.py 是使用模型的函数 
	
## 注意
	可能在加载数据的过程中会报错，没有 content 或者 label 之类的
	这个时候注意检测自己的训练文件和验证文件，一定是某一行没有遵守 label + \t + title + \t + content 的格式
	将这一行删除掉即可
