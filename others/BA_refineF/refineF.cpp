#include <cmath>
#include <cstdio>
#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct F_Residule {
  F_Residule(double p1x_, double p1y_, double p2x_, double p2y_)
      : p1x(p1x_), p1y(p1y_), p2x(p2x_), p2y(p2y_) {}

  template <typename T>
  bool operator()(const T* const F,
                  T* residuals) const {
    T* lr[3];
    lr[0] = F[0]*p1x+F[1]*p1y+F[2];
    lr[1] = F[3]*p1x+F[4]*p1y+F[5];
    lr[2] = F[6]*p1x+F[7]*p1y+F[8];


    // The error is the difference between the predicted and observed position.
    residuals = p2x*lr[0]+p2y*lr[1]+lr[2];

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double ob1_x, const double ob1_y,
                                     const double ob2_x, const double ob2_y) {
    return (new ceres::AutoDiffCostFunction<F_Residule, 1, 9>(
                new F_Residule(ob1_x, ob1_y, ob2_x, ob2_y)));
  }

  double p1x;
  double p2x;
  double p1y;
  double p2y;  
};

int refineF(cv::Mat & F, std::vector<Point2f> pix1, std::vector<Point2f> pix2) {

  double F_refine[9];
  F_refine[0] = F.at(0,0);
  F_refine[1] = F.at(0,1);
  F_refine[2] = F.at(0,2);
  F_refine[3] = F.at(1,0);
  F_refine[4] = F.at(1,1);
  F_refine[5] = F.at(1,2);  
  F_refine[6] = F.at(2,0);
  F_refine[7] = F.at(2,1);
  F_refine[8] = F.at(2,2);  

  vector<int> temp1x, temp1y, temp2x, temp2y;
  for(int i = 0; i<pix1.size(); i++){
      temp1x.push_back(pix1[i].x);
      temp1y.push_back(pix1[i].y);
      temp2x.push_back(pix2[i].x);
      temp2y.push_back(pix2[i].y);
  }
  double* p1x = &temp1x[0];
  double* p1y = &temp1y[0];
  double* p2x = &temp2x[0];
  double* p2y = &temp2y[0];


  ceres::Problem problem;
  for (int i = 0; i < p1x.size(); ++i) {
    ceres::CostFunction* cost_function =
        F_Residule::Create(p1x[i], p1y[i], p2x[i], p2y[i]);
    problem.AddResidualBlock(cost_function,
                             NULL /* squared loss */,
                             &F_refine);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  F.at(0,0) = F_refine[0];
  F.at(0,1) = F_refine[1];
  F.at(0,2) = F_refine[2];
  F.at(1,0) = F_refine[3];
  F.at(1,1) = F_refine[4];
  F.at(1,2) = F_refine[5];  
  F.at(2,0) = F_refine[6];
  F.at(2,1) = F_refine[7];
  F.at(2,2) = F_refine[8];

  return 0;
}
