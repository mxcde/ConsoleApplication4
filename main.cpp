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
#define k 4 //�ص���Ŀ 
using namespace std;
//���Ԫ���������Ϣ 
typedef vector<double> Tuple;//�洢ÿ�����ݼ�¼ 

int dataNum;//���ݼ������ݼ�¼��Ŀ 
int dimNum;//ÿ����¼��ά�� 

		   //��������Ԫ����ŷ������� 
double getDistXY(const Tuple& t1, const Tuple& t2)
{
	double sum = 0;
	for (int i = 1; i <= dimNum; ++i)
	{
		sum += (t1[i] - t2[i]) * (t1[i] - t2[i]);
	}
	return sqrt(sum);
}

//�������ģ�������ǰԪ�������ĸ��� 
int clusterOfTuple(Tuple means[], const Tuple& tuple) {
	double dist = getDistXY(means[0], tuple);
	double tmp;
	int label = 0;//��ʾ������һ���� 
	for (int i = 1; i < k; i++) {
		tmp = getDistXY(means[i], tuple);
		if (tmp < dist) { dist = tmp; label = i; }
	}
	return label;
}
//��ø����ؼ���ƽ����� 
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
//��õ�ǰ�صľ�ֵ�����ģ� 
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
		cout << "��" << lable + 1 << "���أ�" << endl;
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
	vector<Tuple> clusters[k];//k���� 
	Tuple means[k];//k�����ĵ� 
	int i = 0;
	//һ��ʼ���ѡȡk����¼��ֵ��Ϊk���ص����ģ���ֵ�� 
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
	//����Ĭ�ϵ����ĸ��ظ�ֵ 
	for (i = 0; i != tuples.size(); ++i) {
		lable = clusterOfTuple(means, tuples[i]);
		clusters[lable].push_back(tuples[i]);
	}
	double oldVar = -1;
	double newVar = getVar(clusters, means);
	cout << "��ʼ�ĵ��������ƽ����Ϊ��" << newVar << endl;
	int t = 0;
	while (abs(newVar - oldVar) >= 1) //���¾ɺ���ֵ����1��׼����ֵ���������Ա仯ʱ���㷨��ֹ 
	{
		cout << "�� " << ++t << " �ε�����ʼ��" << endl;
		for (i = 0; i < k; i++) //����ÿ���ص����ĵ� 
		{
			means[i] = getMeans(clusters[i]);
		}
		oldVar = newVar;
		newVar = getVar(clusters, means); //�����µ�׼����ֵ 
		for (i = 0; i < k; i++) //���ÿ���� 
		{
			clusters[i].clear();
		}
		//�����µ����Ļ���µĴ� 
		for (i = 0; i != tuples.size(); ++i) {
			lable = clusterOfTuple(means, tuples[i]);
			clusters[lable].push_back(tuples[i]);
		}
		cout << "�˴ε���֮����������ƽ����Ϊ��" << newVar << endl;
	}

	cout << "The result is:\n";
	print(clusters);
}
int main() {

	char fname[256];
	cout << "�����������ݵ��ļ����� ";
	cin >> fname;
	cout << endl << " ����������: ά�� ������Ŀ" << endl;
	cout << endl << " ά��dimNum: ";
	cin >> dimNum;
	cout << endl << " ������ĿdataNum: ";
	cin >> dataNum;
	ifstream infile(fname);
	if (!infile) {
		cout << "���ܴ�������ļ�" << fname << endl;
		return 0;
	}
	vector<Tuple> tuples;
	//���ļ����ж������� 
	for (int i = 0; i < dataNum && !infile.eof(); ++i)
	{
		string str;
		getline(infile, str);
		istringstream istr(str);
		Tuple tuple(dimNum + 1, 0);//��һ��λ�ô�ż�¼��ţ���2��dimNum+1��λ�ô��ʵ��Ԫ�� 
		tuple[0] = i + 1;
		for (int j = 1; j <= dimNum; ++j)
		{
			istr >> tuple[j];
		}
		tuples.push_back(tuple);
	}

	cout << endl << "��ʼ����" << endl;
	KMeans(tuples);
	return 0;
}



#if 0


#include <Windows.h>
using namespace cv;
using namespace std;

int main()
{
	Mat img = imread("mat1.jpg", IMREAD_COLOR); // Դͼ��
	const int k = 8; // �������


	if (img.empty() || k <= 0)
	{
		return -1;
	}

	Mat imgHSV;
	cvtColor(img, imgHSV, CV_RGB2HSV);// ��rgb�ռ�ת��ΪHSV�ռ�

	Mat  imgData(img.rows*img.cols, 1, CV_32FC3);// һ��3ͨ��������
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
//#define WINDOW_1 "����ǰ"  
//#define WINDOW_2 "�����"  
/*

Mat dstIamge(500, 500, CV_8UC3);
Mat A_dstIamge = dstIamge//.clone();

RNG rng(12345); //�����������  

int clusterCount; int sampleCount;
int MIN_SAMPLECOUNT = 400, MIN_SLUSTERCOUNTS = 5;
static void ShowHelpText();
void cluster(int, void*)
{
	Scalar colorTab[] =     //�����ʾ7������ɫ�ģ��������Ҳ�͸�7����ɫ  
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
   //////   rng  ��������� double float int 
	clusterCount = rng.uniform(2, MIN_SLUSTERCOUNTS + 1);//����֮������������  
	sampleCount = rng.uniform(1, MIN_SAMPLECOUNT + 1);//����1��1001��������������Ҳ����һǧ������  
	
	//  CV_32F   float���� C2 ��ͨ��
	/// mat(    ��         ��   mat������)
	Mat points(sampleCount, 1, CV_32FC2), labels;   //��������������ʵ����Ϊ2ͨ������������Ԫ������ΪPoint2f  

	clusterCount = MIN(clusterCount, sampleCount); // ȡ��С��һ��
	Mat centers(clusterCount, 1, points.type());    //�����洢���������ĵ�  

												/* generate random sample from multigaussian distribution */
	/*for (int k = 0; k < clusterCount; k++) // Generate random points  ���������    
	{
		Point center;// Random point coordinate    /// ��ά������
		center.x = rng.uniform(0, dstIamge.cols);
		center.y = rng.uniform(0, dstIamge.rows);
		Mat pointChunk = points.rowRange(k*sampleCount / clusterCount,
			k == clusterCount - 1 ? sampleCount : (k + 1)*sampleCount / clusterCount);   //���һ�������������һ����ƽ�ֵģ�  
																						 //ʣ�µ�һ�ݶ������һ��  
																						 // Each of the classes is the same variance, but the mean is different.    
		rng.fill(pointChunk, CV_RAND_NORMAL, Scalar(center.x, center.y),//the mean  
			Scalar(dstIamge.cols*0.05, dstIamge.rows*0.05)); //the same variance  
	}
	randShuffle(points, 1, &rng);   //��ΪҪ���࣬�������������points����ĵ㣬ע��points��pointChunk�ǹ������ݵġ�  
	dstIamge = Scalar::all(0);

	for (int i = 0; i < sampleCount; i++)
	{

		Point p = points.at<Point2f>(i);// Coordinates of corresponding points    
		circle(A_dstIamge, p, 1, Scalar::all(255), CV_FILLED, CV_AA); ////
	}
	imshow(WINDOW_1, A_dstIamge);

	/*typedef struct CvTermCriteria
	{
		int    type;  /* CV_TERMCRIT_ITER ��CV_TERMCRIT_EPS��ֵ֮һ�����߶��ߵ���� */
//		int    max_iter; /* ���������� */
//		double epsilon; /* ����ľ�ȷ�� */  //��ֹ����  ��ȷ��
	//}
/*	kmeans(points, clusterCount, labels,                          //labels��ʾÿһ����������ı�ǩ����һ����������0��ʼ����������,�Ǵ���.  
		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),//���������������߾�����Ϊ�������������ĸ�����������  
		3,                 //����3��,ȡ�����õ��ǴΣ�  
		KMEANS_PP_CENTERS,//���ʾΪ���ѡȡ��ʼ�����ĵ�,����ĳ�ʼ������PP�ض�������㷨�� 
		                  /*
						  ��ȡֵ��3����������ΪKMEANS_RANDOM_CENTERS�����ʾΪ���ѡȡ��ʼ�����ĵ㣬
						  ���ΪKMEANS_PP_CENTERS���ʾʹ��ĳһ���㷨��ȷ����ʼ����ĵ㣻
						  ���ΪKMEANS_USE_INITIAL_LABELS�����ʾʹ���û��Զ���ĳ�ʼ�㣬
						  ���������ʱ��attempts����1�������ľ����ʼ������ʹ������ķ�ʽ 
						  */
/*		centers);        //����centers��ʾ���Ǿ��������ĵ��ž���  
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