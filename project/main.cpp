#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>

const float param_13 = 1.0f / 3.0f;
const float param_16116 = 16.0f / 116.0f;
const float Xn = 0.950456f;
const float Yn = 1.0f;
const float Zn = 1.088754f;

using namespace std;
using namespace cv;


float gamma(float x)
{
    return x > 0.04045 ? powf((x + 0.055f) / 1.055f, 2.4f) : (x / 12.92);
}

float gamma_XYZ2RGB(float x)
{
    return x > 0.0031308 ? (1.055f * powf(x, (1 / 2.4f)) - 0.055) : (x * 12.92);
}


void XYZ2RGB(float X, float Y, float Z, int *R, int *G, int *B)
{
    float RR, GG, BB;
    RR = 3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
    GG = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
    BB = 0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;

    RR = gamma_XYZ2RGB(RR);
    GG = gamma_XYZ2RGB(GG);
    BB = gamma_XYZ2RGB(BB);

    RR = int(RR * 255.0 + 0.5);
    GG = int(GG * 255.0 + 0.5);
    BB = int(BB * 255.0 + 0.5);

    *R = RR;
    *G = GG;
    *B = BB;
}

void Lab2XYZ(float L, float a, float b, float *X, float *Y, float *Z)
{
    float fX, fY, fZ;

    fY = (L + 16.0f) / 116.0;
    fX = a / 500.0f + fY;
    fZ = fY - b / 200.0f;

    if (powf(fY, 3.0) > 0.008856)
        *Y = powf(fY, 3.0);
    else
        *Y = (fY - param_16116) / 7.787f;

    if (powf(fX, 3) > 0.008856)
        *X = fX * fX * fX;
    else
        *X = (fX - param_16116) / 7.787f;

    if (powf(fZ, 3.0) > 0.008856)
        *Z = fZ * fZ * fZ;
    else
        *Z = (fZ - param_16116) / 7.787f;

    (*X) *= (Xn);
    (*Y) *= (Yn);
    (*Z) *= (Zn);
}

void RGB2XYZ(int R, int G, int B, float *X, float *Y, float *Z)
{
    float RR = gamma((float) R / 255.0f);
    float GG = gamma((float) G / 255.0f);
    float BB = gamma((float) B / 255.0f);

    *X = 0.4124564f * RR + 0.3575761f * GG + 0.1804375f * BB;
    *Y = 0.2126729f * RR + 0.7151522f * GG + 0.0721750f * BB;
    *Z = 0.0193339f * RR + 0.1191920f * GG + 0.9503041f * BB;
}

void XYZ2Lab(float X, float Y, float Z, float *L, float *a, float *b)
{
    float fX, fY, fZ;

    X /= Xn;
    Y /= Yn;
    Z /= Zn;

    if (Y > 0.008856f)
        fY = pow(Y, param_13);
    else
        fY = 7.787f * Y + param_16116;

    *L = 116.0f * fY - 16.0f;
    *L = *L > 0.0f ? *L : 0.0f;

    if (X > 0.008856f)
        fX = pow(X, param_13);
    else
        fX = 7.787f * X + param_16116;

    if (Z > 0.008856)
        fZ = pow(Z, param_13);
    else
        fZ = 7.787f * Z + param_16116;

    *a = 500.0f * (fX - fY);
    *b = 200.0f * (fY - fZ);
}

void RGB2Lab(int R, int G, int B, float *L, float *a, float *b)
{
    float X, Y, Z;
    RGB2XYZ(R, G, B, &X, &Y, &Z);
    XYZ2Lab(X, Y, Z, L, a, b);
}

void Lab2RGB(float L, float a, float b, int *R, int *G, int *B)
{
    float X, Y, Z;
    Lab2XYZ(L, a, b, &X, &Y, &Z);
    XYZ2RGB(X, Y, Z, R, G, B);
}

int main()
{
    Mat raw_image = imread("../pic6.jpg");
    if (raw_image.empty())
    {
        cout << "read error" << endl;
        return 0;
    }
    vector<vector<vector<float>>> image;//x,y,(L,a,b)

    int rows = raw_image.rows;
    int cols = raw_image.cols;
    cout << "rows:" << rows << " cols:" << cols << endl;
    int N = rows * cols;
    //K个超像素
    int K = 150;
    cout << "cluster num:" << K << endl;
    int M = 40;
    //以步距为S的距离划分超像素
    int S = (int) sqrt(N / K);
    cout << "S:" << S << endl;

    //RGB2Lab
    for (int i = 0; i < rows; i++)
    {
        vector<vector<float>> line;
        for (int j = 0; j < cols; j++)
        {
            vector<float> pixel;
            float L;
            float a;
            float b;

            RGB2Lab(raw_image.at<Vec3b>(i, j)[2], raw_image.at<Vec3b>(i, j)[1], raw_image.at<Vec3b>(i, j)[0], &L, &a,
                    &b);
            pixel.push_back(L);
            pixel.push_back(a);
            pixel.push_back(b);

            line.push_back(pixel);
        }
        image.push_back(line);
    }

    cout << "RGB2Lab is finished" << endl;

    //聚类中心，[x y l a b]
    vector<vector<float>> Cluster;

    //生成所有聚类中心
    for (int i = S / 2; i < rows; i += S)
    {
        for (int j = S / 2; j < cols; j += S)
        {
            vector<float> c;
            c.push_back((float) i);
            c.push_back((float) j);
            c.push_back(image[i][j][0]);
            c.push_back(image[i][j][1]);
            c.push_back(image[i][j][2]);

            Cluster.push_back(c);
        }
    }
    int cluster_num = Cluster.size();
    cout << "init cluster is finished" << endl;

    //获得最小梯度值作为新中心点
    for (int c = 0; c < cluster_num; c++)
    {
        int c_row = (int) Cluster[c][0];
        int c_col = (int) Cluster[c][1];
        //梯度以右侧和下侧两个像素点来计算，分别计算Lab三个的梯度来求和
        //需要保证当前点右侧和下侧是存在的点，否则就向左上移动来替代梯度值
        if (c_row + 1 >= rows)
        {
            c_row = rows - 2;
        }
        if (c_col + 1 >= cols)
        {
            c_col = cols - 2;
        }

        float c_gradient =
                image[c_row + 1][c_col][0] + image[c_row][c_col + 1][0] - 2 * image[c_row][c_col][0] +
                image[c_row + 1][c_col][1] + image[c_row][c_col + 1][1] - 2 * image[c_row][c_col][1] +
                image[c_row + 1][c_col][2] + image[c_row][c_col + 1][2] - 2 * image[c_row][c_col][2];

        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                int tmp_row = c_row + i;
                int tmp_col = c_col + j;

                if (tmp_row + 1 >= rows)
                {
                    tmp_row = rows - 2;
                }
                if (tmp_col + 1 >= cols)
                {
                    tmp_col = cols - 2;
                }

                float tmp_gradient =
                        image[tmp_row + 1][tmp_col][0] + image[tmp_row][tmp_col + 1][0] -
                        2 * image[tmp_row][tmp_col][0] + image[tmp_row + 1][tmp_col][1] +
                        image[tmp_row][tmp_col + 1][1] - 2 * image[tmp_row][tmp_col][1] +
                        image[tmp_row + 1][tmp_col][2] + image[tmp_row][tmp_col + 1][2] -
                        2 * image[tmp_row][tmp_col][2];

                if (tmp_gradient < c_gradient)
                {
                    Cluster[c][0] = (float) tmp_row;
                    Cluster[c][1] = (float) tmp_col;
                    Cluster[c][2] = image[tmp_row][tmp_col][0];
                    Cluster[c][3] = image[tmp_row][tmp_col][1];
                    Cluster[c][3] = image[tmp_row][tmp_col][2];
                    c_gradient = tmp_gradient;
                }
            }
        }
    }

    cout << "move cluster is finished";

    //创建一个dis的矩阵for each pixel = ∞
    vector<vector<double>> distance;
    for (int i = 0; i < rows; ++i)
    {
        vector<double> tmp;
        for (int j = 0; j < cols; ++j)
        {
            tmp.push_back(INT_MAX);
        }
        distance.push_back(tmp);
    }

    //创建一个dis的矩阵for each pixel = -1
    vector<vector<int>> label;
    for (int i = 0; i < rows; ++i)
    {
        vector<int> tmp;
        for (int j = 0; j < cols; ++j)
        {
            tmp.push_back(-1);
        }
        label.push_back(tmp);
    }

    //为每一个Cluster创建一个pixel集合
    vector<vector<vector<int>>> pixel(Cluster.size());

    //核心代码部分，迭代计算
    for (int t = 0; t < 10; t++)
    {
        cout << endl << "iteration num:" << t + 1 << "  ";
        //遍历所有的中心点,在2S范围内进行像素搜索
        int c_num = 0;
        for (int c = 0; c < cluster_num; c++)
        {
            if (c - c_num >= (cluster_num / 10))
            {
                cout << "+";
                c_num = c;
            }
            int c_row = (int) Cluster[c][0];
            int c_col = (int) Cluster[c][1];
            float c_L = Cluster[c][2];
            float c_a = Cluster[c][3];
            float c_b = Cluster[c][4];
            for (int i = c_row - 2 * S; i <= c_row + 2 * S; i++)
            {
                if (i < 0 || i >= rows)
                {
                    continue;
                }

                for (int j = c_col - 2 * S; j <= c_col + 2 * S; j++)
                {
                    if (j < 0 || j >= cols)
                    {
                        continue;
                    }

                    float tmp_L = image[i][j][0];
                    float tmp_a = image[i][j][1];
                    float tmp_b = image[i][j][2];

                    double Dc = sqrt((tmp_L - c_L) * (tmp_L - c_L) + (tmp_a - c_a) * (tmp_a - c_a) +
                                     (tmp_b - c_b) * (tmp_b - c_b));
                    double Ds = sqrt((i - c_row) * (i - c_row) + (j - c_col) * (j - c_col));
                    double D = sqrt((Dc / (double) M) * (Dc / (double) M) + (Ds / (double) S) * (Ds / (double) S));

                    if (D < distance[i][j])
                    {
                        if (label[i][j] == -1)
                        {//还没有被标记过
                            label[i][j] = c;

                            vector<int> point;
                            point.push_back(i);
                            point.push_back(j);
                            pixel[c].push_back(point);
                        }
                        else
                        {
                            int old_cluster = label[i][j];
                            vector<vector<int>>::iterator iter;
                            for (iter = pixel[old_cluster].begin(); iter != pixel[old_cluster].end(); iter++)
                            {
                                if ((*iter)[0] == i && (*iter)[1] == j)
                                {
                                    pixel[old_cluster].erase(iter);
                                    break;
                                }
                            }

                            label[i][j] = c;

                            vector<int> point;
                            point.push_back(i);
                            point.push_back(j);
                            pixel[c].push_back(point);
                        }
                        distance[i][j] = D;
                    }
                }
            }
        }

        cout << " start update cluster";

        for (int c = 0; c < Cluster.size(); c++)
        {
            int sum_i = 0;
            int sum_j = 0;
            int number = 0;
            for (int p = 0; p < pixel[c].size(); p++)
            {
                sum_i += pixel[c][p][0];
                sum_j += pixel[c][p][1];
                number++;
            }

            int tmp_i = (int) ((double) sum_i / (double) number);
            int tmp_j = (int) ((double) sum_j / (double) number);

            Cluster[c][0] = (float) tmp_i;
            Cluster[c][1] = (float) tmp_j;
            Cluster[c][2] = image[tmp_i][tmp_j][0];
            Cluster[c][3] = image[tmp_i][tmp_j][1];
            Cluster[c][4] = image[tmp_i][tmp_j][2];
        }
    }

    //导出Lab空间的矩阵
    vector<vector<vector<float>>> out_image = image;//x,y,(L,a,b)
    for (int c = 0; c < Cluster.size(); c++)
    {
        for (int p = 0; p < pixel[c].size(); p++)
        {
            out_image[pixel[c][p][0]][pixel[c][p][1]][0] = Cluster[c][2];
            out_image[pixel[c][p][0]][pixel[c][p][1]][1] = Cluster[c][3];
            out_image[pixel[c][p][0]][pixel[c][p][1]][2] = Cluster[c][4];
        }
        out_image[(int) Cluster[c][0]][(int) Cluster[c][1]][0] = 0;
        out_image[(int) Cluster[c][0]][(int) Cluster[c][1]][1] = 0;
        out_image[(int) Cluster[c][0]][(int) Cluster[c][1]][2] = 0;
    }
    cout << endl << "export image mat finished" << endl;
    Mat print_image = raw_image.clone();
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float L = out_image[i][j][0];
            float a = out_image[i][j][1];
            float b = out_image[i][j][2];

            int R, G, B;
            Lab2RGB(L, a, b, &R, &G, &B);
            Vec3b vec3b;
            vec3b[0] = B;
            vec3b[1] = G;
            vec3b[2] = R;
            print_image.at<Vec3b>(i, j) = vec3b;
        }
    }

    imshow("print_image", print_image);
    waitKey(0);  //暂停，保持图像显示，等待按键结束
    return 0;
}
