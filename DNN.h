#include <algorithm>
#include <functional>
#include <time.h>
#include <numeric>
using namespace std;

#define TESTS 20
#define RANGE 100000
#define SET_RANGE 100
#define NUM_PER_SET_MIN 4000
#define NUM_PER_SET_MAX 5000
#define rint unsigned int
#define range(x) (x).begin(),(x).end()
#define sigmoid(x) (1./(1.+exp(-(x))))
#define CE(a,b) ((a)*log(max(b,1e-12))+(1-(a))*log(max(1-(b),1e-12)))

typedef vector<double> Arr; mt19937 _rd(1061109589);
auto real_rd = std::bind(std::normal_distribution<double>(0., .01), mt19937(998244353));

double Loss, Score;
mt19937 rd(time(0));
int kn = rd() % 30 + 1, T = TESTS, _;

struct P
{
	unsigned int x, y, i;
};

vector<P> Set[32], Homework, Exam;

inline int rf()
{
	int r;
	int s = 0, c;
	for (; !isdigit(c = getchar()); s = c);
	for (r = c ^ 48; isdigit(c = getchar()); (r *= 10) += c ^ 48);
	return s ^ 45 ? r : -r;
}

struct NeuralNetwork
{
	struct Layer
	{
		Arr h, x, d;
		vector<Arr> w;
		int n;

		inline Layer(int _n, int c)
		{
			n = _n;
			x = h = d = Arr(n + 1);
			x[0] = 1;
			w.clear();
			for (int i = 0, j; i <= n; w[i][0] = 0.5, i++)
				for (w.push_back(Arr(c + 1)), j = 1; j <= c; w[i][j] = real_rd(), j++);
		}
	};

	vector<Layer> L;
	Arr Out, Ans, In;	//對外的接口
	int N, kK;
	double eta;			//學習率，代碼中為eta

	inline void AddLayer(int n)
	{
		++N;
		L.emplace_back(n, kK);	//新增Layer空間
		kK = n;
	}

	inline NeuralNetwork(vector<int> Num, double _eta)
	{
		L.clear();
		eta = _eta;
		N = -1;
		kK = 0;
		for (int& j : Num)
			AddLayer(j);
	}

	inline void Run()	//正向傳播
	{
		for (int i = (copy(range(In), ++L[0].x.begin()), 1), j; i <= N; i++)
			for (j = 1; j <= L[i].n; L[i].x[j] = sigmoid(L[i].h[j] = inner_product(range(L[i - 1].x), L[i].w[j].begin(), 0.)), j++);
		Out.resize(kK);

		for (int j = 0; j < kK; Out[j] = exp(L[N].h[j + 1]), j++);
		double S = accumulate(range(Out), 0.);
		for (double& j : Out) j /= S;
	}

	inline int Judge()
	{
		return (int)(max_element(range(Out)) - Out.begin());
	}

	inline double MSELoss()
	{
		double S = 0;
		for (int j; j < kK; pow(Out[j] - Ans[j], 2), j++);
		return (1 / (2 * kK)) * S;
	}

	inline double CELoss()	//交叉熵損失函數(CE)
	{
		double S = 0;
		for (int j = 0; j < kK; S += CE(Ans[j], Out[j]), j++);
		return -S;
	}

	inline void Adjust()	//反向傳播
	{
		for (int j = 1; j <= kK; L[N].d[j] = -eta * (Out[j - 1] - Ans[j - 1]), j++);

		for (int i = N - 1, j, k; i; i--)
			for (fill(range(L[i].d), 0.), j = 1; j <= L[i + 1].n; j++)
				for (k = 0; k <= L[i].n; L[i].d[k] += L[i + 1].d[j] * L[i + 1].w[j][k] * L[i].x[k] * (1 - L[i].x[k]), k++);

		for (int i = N, j, k; i; i--)
			for (j = 1; j <= L[i].n; j++)
				for (k = 0; k <= L[i - 1].n; L[i].w[j][k] += L[i].d[j] * L[i - 1].x[k], k++);
	}
};