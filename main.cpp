#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>



#include <iostream> 
#include <sstream> 
#include <fstream> 
#include <vector> 
#include <math.h> 
#include <stdlib.h> 
#include <Windows.h>
#include <ctime>
#define k 4 //簇的数目 
using namespace std;
//存放元组的属性信息 
typedef vector<double> Tuple;//存储每条数据记录 

int dataNum;//数据集中数据记录数目 
int dimNum;//每条记录的维数 

		   //计算两个元组间的欧几里距离 
double getDistXY(const Tuple& t1, const Tuple& t2)
{
	double sum = 0;
	for (int i = 1; i <= dimNum; ++i)
	{
		sum += (t1[i] - t2[i]) * (t1[i] - t2[i]);
	}
	return sqrt(sum);
}

//根据质心，决定当前元组属于哪个簇 
int clusterOfTuple(Tuple means[], const Tuple& tuple) {
	double dist = getDistXY(means[0], tuple);
	double tmp;
	int label = 0;//标示属于哪一个簇 
	for (int i = 1; i < k; i++) {
		tmp = getDistXY(means[i], tuple);
		if (tmp < dist) { dist = tmp; label = i; }
	}
	return label;
}
//获得给定簇集的平方误差 
double getVar(vector<Tuple> clusters[], Tuple means[]) {
	double var = 0;
	for (int i = 0; i < k; i++)
	{
		vector<Tuple> t = clusters[i];
		for (int j = 0; j < t.size(); j++)
		{
			var += getDistXY(t[j], means[i]);
		}
	}
	//cout<<"sum:"<<sum<<endl; 
	return var;

}
//获得当前簇的均值（质心） 
Tuple getMeans(const vector<Tuple>& cluster) {

	int num = cluster.size();
	Tuple t(dimNum + 1, 0);
	for (int i = 0; i < num; i++)
	{
		for (int j = 1; j <= dimNum; ++j)
		{
			t[j] += cluster[i][j];
		}
	}
	for (int j = 1; j <= dimNum; ++j)
		t[j] /= num;
	return t;
	//cout<<"sum:"<<sum<<endl; 
}

void print(const vector<Tuple> clusters[])
{
	for (int lable = 0; lable < k; lable++)
	{
		cout << "第" << lable + 1 << "个簇：" << endl;
		vector<Tuple> t = clusters[lable];
		for (int i = 0; i < t.size(); i++)
		{
			cout << i + 1 << ".(";
			for (int j = 0; j <= dimNum; ++j)
			{
				cout << t[i][j] << ", ";
			}
			cout << ")\n";
		}
	}
}

void KMeans(vector<Tuple>& tuples) {
	vector<Tuple> clusters[k];//k个簇 
	Tuple means[k];//k个中心点 
	int i = 0;
	//一开始随机选取k条记录的值作为k个簇的质心（均值） 
	srand((unsigned int)time(NULL));
	for (i = 0; i < k;) {
		int iToSelect = rand() % tuples.size();
		if (means[iToSelect].size() == 0)
		{
			for (int j = 0; j <= dimNum; ++j)
			{
				means[i].push_back(tuples[iToSelect][j]);
			}
			++i;
		}
	}
	int lable = 0;
	//根据默认的质心给簇赋值 
	for (i = 0; i != tuples.size(); ++i) {
		lable = clusterOfTuple(means, tuples[i]);
		clusters[lable].push_back(tuples[i]);
	}
	double oldVar = -1;
	double newVar = getVar(clusters, means);
	cout << "初始的的整体误差平方和为：" << newVar << endl;
	int t = 0;
	while (abs(newVar - oldVar) >= 1) //当新旧函数值相差不到1即准则函数值不发生明显变化时，算法终止 
	{
		cout << "第 " << ++t << " 次迭代开始：" << endl;
		for (i = 0; i < k; i++) //更新每个簇的中心点 
		{
			means[i] = getMeans(clusters[i]);
		}
		oldVar = newVar;
		newVar = getVar(clusters, means); //计算新的准则函数值 
		for (i = 0; i < k; i++) //清空每个簇 
		{
			clusters[i].clear();
		}
		//根据新的质心获得新的簇 
		for (i = 0; i != tuples.size(); ++i) {
			lable = clusterOfTuple(means, tuples[i]);
			clusters[lable].push_back(tuples[i]);
		}
		cout << "此次迭代之后的整体误差平方和为：" << newVar << endl;
	}

	cout << "The result is:\n";
	print(clusters);
}
int main() {

	char fname[256];
	cout << "请输入存放数据的文件名： ";
	cin >> fname;
	cout << endl << " 请依次输入: 维数 样本数目" << endl;
	cout << endl << " 维数dimNum: ";
	cin >> dimNum;
	cout << endl << " 样本数目dataNum: ";
	cin >> dataNum;
	ifstream infile(fname);
	if (!infile) {
		cout << "不能打开输入的文件" << fname << endl;
		return 0;
	}
	vector<Tuple> tuples;
	//从文件流中读入数据 
	for (int i = 0; i < dataNum && !infile.eof(); ++i)
	{
		string str;
		getline(infile, str);
		istringstream istr(str);
		Tuple tuple(dimNum + 1, 0);//第一个位置存放记录编号，第2到dimNum+1个位置存放实际元素 
		tuple[0] = i + 1;
		for (int j = 1; j <= dimNum; ++j)
		{
			istr >> tuple[j];
		}
		tuples.push_back(tuple);
	}

	cout << endl << "开始聚类" << endl;
	KMeans(tuples);
	return 0;
}



#if 0


#include <Windows.h>
using namespace cv;
using namespace std;

int main()
{
	Mat img = imread("mat1.jpg", IMREAD_COLOR); // 源图像
	const int k = 8; // 聚类个数


	if (img.empty() || k <= 0)
	{
		return -1;
	}

	Mat imgHSV;
	cvtColor(img, imgHSV, CV_RGB2HSV);// 将rgb空间转换为HSV空间

	Mat  imgData(img.rows*img.cols, 1, CV_32FC3);// 一个3通道的数据
	Mat_<Vec3b>::iterator itImg = imgHSV.begin<Vec3b>();
	Mat_<Vec3f>::iterator itData = imgData.begin<Vec3f>();

	for (; itImg != imgHSV.end<Vec3b>(); ++itImg,++itData)
	{
		*itData = *itImg;
	}

	Mat imgLabel, imgCenter;

	kmeans(imgData, k, imgLabel,
		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
		1, KMEANS_PP_CENTERS,imgCenter);

	Mat imgRes(img.size(), CV_8UC3);
	Mat_<Vec3b>::iterator itRes = imgRes.begin<Vec3b>();
	Mat_<int>::iterator itLabel = imgLabel.begin<int>();

	for (; itLabel != imgLabel.end<int>(); ++itLabel, ++itRes)
	{
		*itRes = imgCenter.at<Vec3f>(*itLabel, 0);
	}

	cvtColor(imgRes, imgRes, CV_HSV2BGR);
	imwrite("out1.jpg", imgRes);

	namedWindow("img", WINDOW_AUTOSIZE);
	imshow("img", img);
	namedWindow("res", WINDOW_AUTOSIZE);
	imshow("res", imgRes);


	waitKey();
	return 0;


}
#endif
//#define WINDOW_1 "聚类前"  
//#define WINDOW_2 "聚类后"  
/*

Mat dstIamge(500, 500, CV_8UC3);
Mat A_dstIamge = dstIamge//.clone();

RNG rng(12345); //随机数产生器  

int clusterCount; int sampleCount;
int MIN_SAMPLECOUNT = 400, MIN_SLUSTERCOUNTS = 5;
static void ShowHelpText();
void cluster(int, void*)
{
	Scalar colorTab[] =     //最多显示7类有颜色的，所以最多也就给7个颜色  
	{
		Scalar(0, 0, 255),
		Scalar(0, 255, 0),
		Scalar(0, 255, 255),

		Scalar(255, 0, 0),
		Scalar(255, 255, 0),
		Scalar(255, 0, 255),
		Scalar(255, 255, 255),

		Scalar(200, 200, 255),
	};
   //////   rng  产生随机数 double float int 
	clusterCount = rng.uniform(2, MIN_SLUSTERCOUNTS + 1);//产生之间的整数个类别！  
	sampleCount = rng.uniform(1, MIN_SAMPLECOUNT + 1);//产生1到1001个整数样本数，也就是一千个样本  
	
	//  CV_32F   float类型 C2 两通道
	/// mat(    行         列   mat的类型)
	Mat points(sampleCount, 1, CV_32FC2), labels;   //产生的样本数，实际上为2通道的列向量，元素类型为Point2f  

	clusterCount = MIN(clusterCount, sampleCount); // 取最小的一个
	Mat centers(clusterCount, 1, points.type());    //用来存储聚类后的中心点  

												/* generate random sample from multigaussian distribution */
	/*for (int k = 0; k < clusterCount; k++) // Generate random points  产生随机点    
	{
		Point center;// Random point coordinate    /// 二维点坐标
		center.x = rng.uniform(0, dstIamge.cols);
		center.y = rng.uniform(0, dstIamge.rows);
		Mat pointChunk = points.rowRange(k*sampleCount / clusterCount,
			k == clusterCount - 1 ? sampleCount : (k + 1)*sampleCount / clusterCount);   //最后一个类的样本数不一定是平分的，  
																						 //剩下的一份都给最后一类  
																						 // Each of the classes is the same variance, but the mean is different.    
		rng.fill(pointChunk, CV_RAND_NORMAL, Scalar(center.x, center.y),//the mean  
			Scalar(dstIamge.cols*0.05, dstIamge.rows*0.05)); //the same variance  
	}
	randShuffle(points, 1, &rng);   //因为要聚类，所以先随机打乱points里面的点，注意points和pointChunk是共用数据的。  
	dstIamge = Scalar::all(0);

	for (int i = 0; i < sampleCount; i++)
	{

		Point p = points.at<Point2f>(i);// Coordinates of corresponding points    
		circle(A_dstIamge, p, 1, Scalar::all(255), CV_FILLED, CV_AA); ////
	}
	imshow(WINDOW_1, A_dstIamge);

	/*typedef struct CvTermCriteria
	{
		int    type;  /* CV_TERMCRIT_ITER 和CV_TERMCRIT_EPS二值之一，或者二者的组合 */
//		int    max_iter; /* 最大迭代次数 */
//		double epsilon; /* 结果的精确性 */  //终止条件  精确度
	//}
/*	kmeans(points, clusterCount, labels,                          //labels表示每一个样本的类的标签，是一个整数，从0开始的索引整数,是簇数.  
		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),//用最大迭代次数或者精度作为迭代条件，看哪个条件先满足  
		3,                 //聚类3次,取结果最好的那次，  
		KMEANS_PP_CENTERS,//则表示为随机选取初始化中心点,聚类的初始化采用PP特定的随机算法。 
		                  /*
						  其取值有3种情况，如果为KMEANS_RANDOM_CENTERS，则表示为随机选取初始化中心点，
						  如果为KMEANS_PP_CENTERS则表示使用某一种算法来确定初始聚类的点；
						  如果为KMEANS_USE_INITIAL_LABELS，则表示使用用户自定义的初始点，
						  但是如果此时的attempts大于1，则后面的聚类初始点依旧使用随机的方式 
						  */
/*		centers);        //参数centers表示的是聚类后的中心点存放矩阵。  
				          // Traverse each point   
	for (int i = 0; i < sampleCount; i++)
	{
		int clusterIdx = labels.at<int>(i);// A label has been completed by clustering    
		Point p = points.at<Point2f>(i);// Coordinates of corresponding points    
		circle(dstIamge, p, 1, colorTab[clusterIdx], CV_FILLED, CV_AA);
	}

	imshow(WINDOW_2, dstIamge);
}
int main()
{
	ShowHelpText();
	while (1)
	{

		namedWindow(WINDOW_1, WINDOW_AUTOSIZE);
		createTrackbar("samleCounts: ", WINDOW_1, &MIN_SAMPLECOUNT, 1000, cluster);
		cluster(0, 0);

		createTrackbar("clusterCounts: ", WINDOW_1, &MIN_SLUSTERCOUNTS, 10, cluster);
		cluster(0, 0);


		char key = (char)waitKey(); //wait forever  
		if (key == 27 || key == 'q' || key == 'Q')
			break;
	}
	return 0;
}
static void ShowHelpText()
{
	cout << "\n\n\n\n\n" << "                    " << " Kmeans algorithm completion process  ! " << endl;
}

/*
int main()
{
	/*
	Mat img = imread("20140225205530656.jpg");
	if (img.empty())
	{
		cout << "error";
		return -1;
	}
	imshow("20140225205530656", img);
	waitKey();
	
	
	char* pst = "img";
	//IplImage *pimage = cvLoadImage("20140225205530656.jpg", CV_LOAD_IMAGE_UNCHANGED);
	Mat pimage = imread("20140225205530656.jpg", CV_LOAD_IMAGE_ANYDEPTH);

	namedWindow(pst, CV_WINDOW_AUTOSIZE);
	imshow(pst, pimage);
	
	waitKey(0);
	destroyWindow(pst);
	//rel(&pimage);

	//getchar();
	return 0;
}*/