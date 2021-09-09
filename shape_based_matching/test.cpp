#include "line2Dup.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
using namespace std;
using namespace cv;

static std::string prefix = "/home/huangyf/work/AMR/AMR/shape_based_matching/test/";

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};
// NMS, got from cv::dnn so we don't need opencv contrib
// just collapse it
namespace  cv_dnn {
namespace
{

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2)
{
    return pair1.first > pair2.first;
}

} // namespace

inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                      std::vector<std::pair<float, int> >& score_index_vec)
{
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > threshold)
        {
            score_index_vec.push_back(std::make_pair(scores[i], i));
        }
    }
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend<int>);
    if (top_k > 0 && top_k < (int)score_index_vec.size())
    {
        score_index_vec.resize(top_k);
    }
}

template <typename BoxType>
inline void NMSFast_(const std::vector<BoxType>& bboxes,
      const std::vector<float>& scores, const float score_threshold,
      const float nms_threshold, const float eta, const int top_k,
      std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
{
    CV_Assert(bboxes.size() == scores.size());
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int)indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
            keep = overlap <= adaptive_threshold;
        }
        if (keep)
            indices.push_back(idx);
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
          adaptive_threshold *= eta;
        }
    }
}


// copied from opencv 3.4, not exist in 3.0
template<typename _Tp> static inline
double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
    _Tp Aa = a.area();
    _Tp Ab = b.area();

    if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
        // jaccard_index = 1 -> distance = 0
        return 0.0;
    }

    double Aab = (a & b).area();
    // distance = 1 - jaccard_index
    return 1.0 - Aab / (Aa + Ab - Aab);
}

template <typename T>
static inline float rectOverlap(const T& a, const T& b)
{
    return 1.f - static_cast<float>(jaccardDistance__(a, b));
}

void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                          const float score_threshold, const float nms_threshold,
                          std::vector<int>& indices, const float eta=1, const int top_k=0)
{
    NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
}

}

bool AMR(string mode, string type_code, string img_name, int& center_x, int& center_y, int& center_r, int& center_angle){
    line2Dup::Detector detector(30, {4, 8});

    if(mode == "train"){
        Mat img = imread(img_name);
        assert(!img.empty() && "check your img path");
        Mat mask = Mat(img.size(), CV_8UC1, {255});

        shape_based_matching::shapeInfo_producer shapes(img, mask);
        shapes.angle_range = {0, 360};
        shapes.angle_step = 1;
        shapes.produce_infos();
        std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
        string class_id = type_code;
        for(auto& info: shapes.infos){
            //imshow("train", shapes.src_of(info));
            //waitKey(1);

            std::cout << "\ninfo.angle: " << info.angle << std::endl;
            int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info));
            std::cout << "templ_id: " << templ_id << std::endl;
            if(templ_id != -1){
                infos_have_templ.push_back(info);
            }
        }
        detector.writeClasses(prefix+"AMR/%s_templ.yaml");
        shapes.save_infos(infos_have_templ, prefix + "AMR/"+type_code+"_info.yaml");
        std::cout << "train end" << std::endl << std::endl;
    }else if(mode=="test"){
        std::vector<std::string> ids;
		std::cout << "type code " << type_code<<endl;
        ids.push_back(type_code);
        detector.readClasses(ids, prefix+"AMR/%s_templ.yaml");

        Mat test_img = imread(img_name);
        assert(!test_img.empty() && "check your img path");

        auto infos = shape_based_matching::shapeInfo_producer::load_infos(prefix + "AMR/"+type_code+"_info.yaml");

        // cvtColor(test_img, test_img, CV_BGR2GRAY);

        int stride = 16;
        int n = test_img.rows/stride;
        int m = test_img.cols/stride;
        Rect roi(0, 0, stride*m , stride*n);

        test_img = test_img(roi).clone();

        Timer timer;
        auto matches = detector.match(test_img, 90, ids);
        timer.out();

        std::cout << "matches.size(): " << matches.size() << std::endl;
        size_t top5 = 500;
        if(top5>matches.size()) top5=matches.size();

        vector<Rect> boxes;
        vector<float> scores;
        vector<int> idxs;
        for(auto match: matches){
            Rect box;
            box.x = match.x;
            box.y = match.y;

            auto templ = detector.getTemplates(type_code,
                                               match.template_id);

            box.width = templ[0].width;
            box.height = templ[0].height;
            boxes.push_back(box);
            scores.push_back(match.similarity);
        }
        cv_dnn::NMSBoxes(boxes, scores, 0, 0.5f, idxs);

		center_x = 0;
		center_y = 0;
		center_r = 0;
		center_angle = 0;
		const int y0 = test_img.rows/2;
		const int x0 = test_img.cols/2;
		int min_d = 10000;
		int min_index = -1;
        for(auto idx: idxs){
            auto match = matches[idx];
            auto templ = detector.getTemplates(type_code,
                                               match.template_id);

            int x = templ[0].width + match.x;
            int y = templ[0].height + match.y;
            int r = templ[0].width/2;
			int temp_x = (match.x + templ[0].width/2);
			int temp_y = (match.y + templ[0].height/2);
			int d =fabs(temp_x-x0) + fabs(temp_y-y0);
			if ( d < min_d)
			{
				min_d = d;
				min_index = idx;
				center_x = temp_x;
				center_y = temp_y;
				center_r = r;
				center_angle = infos[match.template_id].angle;
			}
			std::cout << "index " << idx << " coord " << center_x << "," << center_y << "similarity " << matches[idx].similarity 
				<< "r " << r << "angle " << infos[match.template_id].angle << "distance:" << d << "center: " << x0 << "," << y0 << endl;
            cv::Vec3b randColor;
            randColor[0] = rand()%155 + 100;
            randColor[1] = rand()%155 + 100;
            randColor[2] = rand()%155 + 100;

            for(int i=0; i<templ[0].features.size(); i++){
                auto feat = templ[0].features[i];
                cv::circle(test_img, {feat.x+match.x, feat.y+match.y}, 2, randColor, -1);
            }

            cv::putText(test_img, to_string(int(round(match.similarity))),
                        Point(match.x+r-10, match.y-3), FONT_HERSHEY_PLAIN, 2, randColor);
            cv::rectangle(test_img, {match.x, match.y}, {x, y}, randColor, 2);

            std::cout << "\nmatch.template_id: " << match.template_id << std::endl;
            std::cout << "match.similarity: " << match.similarity << std::endl;
        }
		std::cout << "mathed count: " <<idxs.size() << "min index " << min_index << endl;
		imwrite(prefix+"AMR/test_result.png", test_img);
    }
	else
	{
            std::cout << "error parameter mode " << std::endl;
	}


	return true;
}

int main(int argc,char **argv){
    if (argc != 4)
	    std::cout << "input should with 1 parameter of image path" << std::endl;
    else
	{
	    int x,y,r, angle;
		AMR(argv[1], argv[2], argv[3], x, y, r, angle);
		std::cout << x << "," << y << "," << r << "," << angle << endl;
	}
    return 0;
}
