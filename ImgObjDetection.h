#include <string>
#include <vector>
using namespace std;

typedef struct
{
	string modelWeights;
	string modelConfiguration;
	string classesFile;
}Modelstr;

class ImgObjDetection
{
private:
	void TurnTo(unsigned char* in, unsigned char* out, int w, int h);

public:
	ImgObjDetection();
	~ImgObjDetection();

	void DetectImage(unsigned char* inptr, int w, int h, Modelstr str);
	void DetectVideo(string vstr, string ostr, Modelstr str);
	void DetectStreaming(Modelstr str);
};