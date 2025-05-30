/**
 * @file ITCModel.cpp
 * @brief Source of the Boussinesq thermal convection in a sphere
 * (Toroidal/Poloidal formulation)
 */

// System includes
//

// Project includes
//
#include "Model/Boussinesq/Sphere/TC/ITCModel.hpp"
#include "Model/Boussinesq/Sphere/TC/Momentum.hpp"
#include "Model/Boussinesq/Sphere/TC/Transport.hpp"
#include "Model/Boussinesq/Sphere/TC/gitHash.hpp"
#include "QuICC/Enums/FieldIds.hpp"
#include "QuICC/Generator/States/RandomScalarState.hpp"
#include "QuICC/Generator/States/RandomVectorState.hpp"
#include "QuICC/Generator/States/SphereExactScalarState.hpp"
#include "QuICC/Generator/States/SphereExactVectorState.hpp"
#include "QuICC/Generator/Visualizers/ScalarFieldVisualizer.hpp"
#include "QuICC/Generator/Visualizers/VectorFieldVisualizer.hpp"
#include "QuICC/Io/Variable/SphereAngularMomentumWriter.hpp"
#include "QuICC/Io/Variable/SphereMaxAbsoluteFieldValueWriter.hpp"
#include "QuICC/Io/Variable/SphereNusseltWriter.hpp"
#include "QuICC/Io/Variable/SphereScalarEnergyWriter.hpp"
#include "QuICC/Io/Variable/SphereScalarLSpectrumWriter.hpp"
#include "QuICC/Io/Variable/SphereScalarMSpectrumWriter.hpp"
#include "QuICC/Io/Variable/SphereScalarMeanWriter.hpp"
#include "QuICC/Io/Variable/SphereScalarNSpectrumWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolEnergyWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolEnstrophyLSpectrumWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolEnstrophyMSpectrumWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolEnstrophyWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolLSpectrumWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolMSpectrumWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolModeSpectrumWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolNSpectrumWriter.hpp"
#include "QuICC/Io/Variable/StateFileReader.hpp"
#include "QuICC/Io/Variable/StateFileWriter.hpp"
#include "QuICC/Io/Variable/VisualizationFileWriter.hpp"
#include "QuICC/NonDimensional/Prandtl.hpp"
#include "QuICC/NonDimensional/Rayleigh.hpp"
#include "QuICC/PhysicalNames/Temperature.hpp"
#include "QuICC/PhysicalNames/Velocity.hpp"
#include "QuICC/SpectralKernels/MakeRandom.hpp"

namespace QuICC {

namespace Model {

namespace Boussinesq {

namespace Sphere {

namespace TC {

VectorFormulation::Id ITCModel::SchemeFormulation()
{
   return VectorFormulation::TORPOL;
}

std::string ITCModel::version() const
{
   return std::string(gitHash);
}

void ITCModel::addEquations(SharedSimulation spSim)
{
   // Add transport equation
   spSim->addEquation<Equations::Boussinesq::Sphere::TC::Transport>(
      this->spBackend());

   // Add Navier-Stokes equation
   spSim->addEquation<Equations::Boussinesq::Sphere::TC::Momentum>(
      this->spBackend());

   #ifdef QUICC_USE_MLIR_GRAPH
   // Add Graph
   std::string graphStr = R"mlir(
// type aliases
!real = tensor<?x?x?xf64>
!complex = tensor<?x?x?xcomplex<f64>>

func.func private @bwdScalar(%S: !complex) -> !real {
    // backward scalar path
    %S1 = quiccir.jw.prj %S : !complex -> !complex attributes{kind = "P"}
    %S1T = quiccir.transpose %S1 permutation = [1, 2, 0] : !complex -> !complex
    %S2 = quiccir.al.prj %S1T : !complex -> !complex attributes{kind = "P"}
    %S2T = quiccir.transpose %S2 permutation = [1, 2, 0] : !complex -> !complex
    %S3 = quiccir.fr.prj %S2T : !complex -> !real attributes{kind = "P"}
    return %S3 : !real
}

func.func private @bwdGradScalar(%S: !complex) -> (!real, !real, !real) {
    // grad R
    %SdR1 = quiccir.jw.prj %S : !complex -> !complex attributes{kind = "D1"}
    %SdR1T = quiccir.transpose %SdR1 permutation = [1, 2, 0] : !complex -> !complex
    %SdR2 = quiccir.al.prj %SdR1T : !complex -> !complex attributes{kind = "P"}
    %SdR2T = quiccir.transpose %SdR2 permutation = [1, 2, 0] : !complex -> !complex
    %SdR = quiccir.fr.prj %SdR2T : !complex -> !real attributes{kind = "P"}
    // grad Theta
    %SdTh1 = quiccir.jw.prj %S : !complex -> !complex attributes{kind = "DivR1_Zero"}
    %SdTh1T = quiccir.transpose %SdTh1 permutation = [1, 2, 0] : !complex -> !complex
    %SdTh2 = quiccir.al.prj %SdTh1T : !complex -> !complex attributes{kind = "D1"}
    %SdTh2T = quiccir.transpose %SdTh2 permutation = [1, 2, 0] : !complex -> !complex
    %SdTh = quiccir.fr.prj %SdTh2T : !complex -> !real attributes{kind = "P"}
    // grad Phi
    %SdPh1 = quiccir.jw.prj %S : !complex -> !complex attributes{kind = "DivR1_Zero"}
    %SdPh1T = quiccir.transpose %SdPh1 permutation = [1, 2, 0] : !complex -> !complex
    %SdPh2 = quiccir.al.prj %SdPh1T : !complex -> !complex attributes{kind = "DivS1Dp"}
    %SdPh2T = quiccir.transpose %SdPh2 permutation = [1, 2, 0] : !complex -> !complex
    %SdPh = quiccir.fr.prj %SdPh2T : !complex -> !real attributes{kind = "P"}
    return %SdR, %SdTh, %SdPh : !real, !real, !real
}

func.func private @bwdVector(%Tor: !complex, %Pol: !complex) -> (!real, !real, !real) {
    // R
    %PolR1 = quiccir.jw.prj %Pol : !complex -> !complex attributes{kind = "DivR1_Zero"}
    %PolR1T = quiccir.transpose %PolR1 permutation = [1, 2, 0] : !complex -> !complex
    %PolR2 = quiccir.al.prj %PolR1T : !complex -> !complex attributes{kind = "Ll"}
    %PolR2T = quiccir.transpose %PolR2 permutation = [1, 2, 0] : !complex -> !complex
    %R = quiccir.fr.prj %PolR2T : !complex -> !real attributes{kind = "P"}
    // Theta
    %TorTh1 = quiccir.jw.prj %Tor : !complex -> !complex attributes{kind = "P"}
    %TorTh1T = quiccir.transpose %TorTh1 permutation = [1, 2, 0] : !complex -> !complex
    %TorTh2 = quiccir.al.prj %TorTh1T : !complex -> !complex attributes{kind = "DivS1Dp"}
    %TorTh2T = quiccir.transpose %TorTh2 permutation = [1, 2, 0] : !complex -> !complex
    %TorTh3 = quiccir.fr.prj %TorTh2T : !complex -> !real attributes{kind = "P"}
    //
    %PolTh1 = quiccir.jw.prj %Pol : !complex -> !complex attributes{kind = "DivR1D1R1_Zero"}
    %PolTh1T = quiccir.transpose %PolTh1 permutation = [1, 2, 0] : !complex -> !complex
    %PolTh2 = quiccir.al.prj %PolTh1T : !complex -> !complex attributes{kind = "D1"}
    %PolTh2T = quiccir.transpose %PolTh2 permutation = [1, 2, 0] : !complex -> !complex
    %PolTh3 = quiccir.fr.prj %PolTh2T : !complex -> !real attributes{kind = "P"}
    //
    %Theta = quiccir.add %TorTh3, %PolTh3 : !real, !real -> !real
    // Phi
    %TorPh1 = quiccir.jw.prj %Tor : !complex -> !complex attributes{kind = "P"}
    %TorPh1T = quiccir.transpose %TorPh1 permutation = [1, 2, 0] : !complex -> !complex
    %TorPh2 = quiccir.al.prj %TorPh1T : !complex -> !complex attributes{kind = "D1"}
    %TorPh2T = quiccir.transpose %TorPh2 permutation = [1, 2, 0] : !complex -> !complex
    %TorPh3 = quiccir.fr.prj %TorPh2T : !complex -> !real attributes{kind = "P"}
    //
    %PolPh1 = quiccir.jw.prj %Pol : !complex -> !complex attributes{kind = "DivR1D1R1_Zero"}
    %PolPh1T = quiccir.transpose %PolPh1 permutation = [1, 2, 0] : !complex -> !complex
    %PolPh2 = quiccir.al.prj %PolPh1T : !complex -> !complex attributes{kind = "DivS1Dp"}
    %PolPh2T = quiccir.transpose %PolPh2 permutation = [1, 2, 0] : !complex -> !complex
    %PolPh3 = quiccir.fr.prj %PolPh2T : !complex -> !real attributes{kind = "P"}
    //
    %Phi = quiccir.sub %PolPh3, %TorPh3 : !real, !real -> !real
    return %R, %Theta, %Phi : !real, !real, !real
}

func.func private @bwdCurl(%Tor: !complex, %Pol: !complex) -> (!real, !real, !real) {
    // R
    %TorR1 = quiccir.jw.prj %Tor : !complex -> !complex attributes{kind = "DivR1_Zero"}
    %TorR1T = quiccir.transpose %TorR1 permutation = [1, 2, 0] : !complex -> !complex
    %TorR2 = quiccir.al.prj %TorR1T : !complex -> !complex attributes{kind = "Ll"}
    %TorR2T = quiccir.transpose %TorR2 permutation = [1, 2, 0] : !complex -> !complex
    %R = quiccir.fr.prj %TorR2T : !complex -> !real attributes{kind = "P"}
    // Theta
    %TorTh1 = quiccir.jw.prj %Tor : !complex -> !complex attributes{kind = "DivR1D1R1_Zero"}
    %TorTh1T = quiccir.transpose %TorTh1 permutation = [1, 2, 0] : !complex -> !complex
    %TorTh2 = quiccir.al.prj %TorTh1T : !complex -> !complex attributes{kind = "D1"}
    %TorTh2T = quiccir.transpose %TorTh2 permutation = [1, 2, 0] : !complex -> !complex
    %TorTh3 = quiccir.fr.prj %TorTh2T : !complex -> !real attributes{kind = "P"}
    //
    %PolTh1 = quiccir.jw.prj %Pol : !complex -> !complex attributes{kind = "SphLapl"}
    %PolTh1T = quiccir.transpose %PolTh1 permutation = [1, 2, 0] : !complex -> !complex
    %PolTh2 = quiccir.al.prj %PolTh1T : !complex -> !complex attributes{kind = "DivS1Dp"}
    %PolTh2T = quiccir.transpose %PolTh2 permutation = [1, 2, 0] : !complex -> !complex
    %PolTh3 = quiccir.fr.prj %PolTh2T : !complex -> !real attributes{kind = "P"}
    //
    %Theta = quiccir.sub %TorTh3, %PolTh3 : !real, !real -> !real
    // Phi
    %TorPh1 = quiccir.jw.prj %Tor : !complex -> !complex attributes{kind = "DivR1D1R1_Zero"}
    %TorPh1T = quiccir.transpose %TorPh1 permutation = [1, 2, 0] : !complex -> !complex
    %TorPh2 = quiccir.al.prj %TorPh1T : !complex -> !complex attributes{kind = "DivS1Dp"}
    %TorPh2T = quiccir.transpose %TorPh2 permutation = [1, 2, 0] : !complex -> !complex
    %TorPh3 = quiccir.fr.prj %TorPh2T : !complex -> !real attributes{kind = "P"}
    //
    %PolPh1 = quiccir.jw.prj %Pol : !complex -> !complex attributes{kind = "SphLapl"}
    %PolPh1T = quiccir.transpose %PolPh1 permutation = [1, 2, 0] : !complex -> !complex
    %PolPh2 = quiccir.al.prj %PolPh1T : !complex -> !complex attributes{kind = "D1"}
    %PolPh2T = quiccir.transpose %PolPh2 permutation = [1, 2, 0] : !complex -> !complex
    %PolPh3 = quiccir.fr.prj %PolPh2T : !complex -> !real attributes{kind = "P"}
    //
    %Phi = quiccir.add %PolPh3, %TorPh3 : !real, !real -> !real
    return %R, %Theta, %Phi : !real, !real, !real
}

func.func private @fwdScalar(%S: !real) -> !complex {
    // forward scalar Nl path
    %S1 = quiccir.fr.int %S : !real -> !complex attributes{kind = "P"}
    %S1T = quiccir.transpose %S1 permutation = [2, 0, 1] : !complex -> !complex
    %S2 = quiccir.al.int %S1T : !complex -> !complex attributes{kind = "P"}
    %S2T = quiccir.transpose %S2 permutation = [2, 0, 1] : !complex -> !complex
    %S3 = quiccir.jw.int %S2T : !complex -> !complex attributes{kind = "I2"}

    return %S3 : !complex
}

func.func private @fwdVector(%R: !real, %Theta: !real, %Phi: !real) -> (!complex, !complex){
    // Tor
    %ThetaTor1 = quiccir.fr.int %Theta : !real -> !complex attributes{kind = "P"}
    %ThetaTor1T = quiccir.transpose %ThetaTor1 permutation = [2, 0, 1] : !complex -> !complex
    %ThetaTor2 = quiccir.al.int %ThetaTor1T : !complex -> !complex attributes{kind = "DivLlDivS1Dp"}
    %ThetaTor2T = quiccir.transpose %ThetaTor2 permutation = [2, 0, 1] : !complex -> !complex
    %ThetaTor3 = quiccir.jw.int %ThetaTor2T : !complex -> !complex attributes{kind = "I2_Zero"}
    //
    %PhiTor1 = quiccir.fr.int %Phi : !real -> !complex attributes{kind = "P"}
    %PhiTor1T = quiccir.transpose %PhiTor1 permutation = [2, 0, 1] : !complex -> !complex
    %PhiTor2 = quiccir.al.int %PhiTor1T : !complex -> !complex attributes{kind = "DivLlD1"}
    %PhiTor2T = quiccir.transpose %PhiTor2 permutation = [2, 0, 1] : !complex -> !complex
    %PhiTor3 = quiccir.jw.int %PhiTor2T : !complex -> !complex attributes{kind = "I2_Zero"}
    //
    %Tor = quiccir.sub %ThetaTor3, %PhiTor3 : !complex, !complex -> !complex

    // Pol
    %RPol1 = quiccir.fr.int %R : !real -> !complex attributes{kind = "P"}
    %RPol1T = quiccir.transpose %RPol1 permutation = [2, 0, 1] : !complex -> !complex
    %RPol2 = quiccir.al.int %RPol1T : !complex -> !complex attributes{kind = "P"}
    %RPol2T = quiccir.transpose %RPol2 permutation = [2, 0, 1] : !complex -> !complex
    %RPol3 = quiccir.jw.int %RPol2T : !complex -> !complex attributes{kind = "I4DivR1_Zero"}
    //
    %ThetaPol1 = quiccir.fr.int %Theta : !real -> !complex attributes{kind = "P"}
    %ThetaPol1T = quiccir.transpose %ThetaPol1 permutation = [2, 0, 1] : !complex -> !complex
    %ThetaPol2 = quiccir.al.int %ThetaPol1T : !complex -> !complex attributes{kind = "DivLlD1"}
    %ThetaPol2T = quiccir.transpose %ThetaPol2 permutation = [2, 0, 1] : !complex -> !complex
    %ThetaPol3 = quiccir.jw.int %ThetaPol2T : !complex -> !complex attributes{kind = "I4DivR1D1R1_Zero"}
    //
    %PhiPol1 = quiccir.fr.int %Phi : !real -> !complex attributes{kind = "P"}
    %PhiPol1T = quiccir.transpose %PhiPol1 permutation = [2, 0, 1] : !complex -> !complex
    %PhiPol2 = quiccir.al.int %PhiPol1T : !complex -> !complex attributes{kind = "DivLlDivS1Dp"}
    %PhiPol2T = quiccir.transpose %PhiPol2 permutation = [2, 0, 1] : !complex -> !complex
    %PhiPol3 = quiccir.jw.int %PhiPol2T : !complex -> !complex attributes{kind = "I4DivR1D1R1_Zero"}
    //
    %tmp = quiccir.add %ThetaPol3, %PhiPol3 : !complex, !complex -> !complex
    %Pol = quiccir.sub %tmp, %RPol3 : !complex, !complex -> !complex
    return %Tor, %Pol : !complex, !complex
}

func.func private @nlScalar(%UR: !real, %UTheta: !real, %UPhi: !real,
    %TdR: !real, %TdTheta: !real, %TdPhi: !real) -> !real {
    // U dot grad T
    %DotT = quiccir.dot(%UR, %UTheta, %UPhi), (%TdR, %TdTheta, %TdPhi) :
        (!real, !real, !real), (!real, !real, !real) ->
        !real
        attributes{kind = "transport"}
    // U dot R
    %DotR = quiccir.mul.const %UR : !real -> !real attributes{kind = "transport"}
    %TPhysNl = quiccir.sub %DotT, %DotR : !real, !real -> !real
    return %TPhysNl : !real
}

func.func private @nlVector(%UR: !real, %UTheta: !real, %UPhi: !real,
    %CurlR: !real, %CurlTheta: !real, %CurlPhi: !real, %T: !real) -> (!real, !real, !real) {
    // Cross
    %Cross:3 = quiccir.cross(%CurlR, %CurlTheta, %CurlPhi), (%UR, %UTheta, %UPhi) :
        (!real, !real, !real), (!real, !real, !real) ->
        (!real, !real, !real)
        attributes{kind = "inertia"}
    // Add buoyancy
    %Buoy = quiccir.mul.const %T : !real -> !real attributes{kind = "buoyancy"}
    %RNl = quiccir.sub %Cross#0, %Buoy : !real, !real -> !real
    return %RNl, %Cross#1, %Cross#2 : !real, !real, !real
}


func.func @entry(%T: !complex, %Tor: !complex, %Pol: !complex) -> (!complex, !complex, !complex, !real, !real, !real) {
    %TPhys = call @bwdScalar(%T) : (!complex) -> !real
    %TGrad:3 = call @bwdGradScalar(%T) : (!complex) -> (!real, !real, !real)
    %Vel:3 = call @bwdVector(%Tor, %Pol) : (!complex, !complex) -> (!real, !real, !real)
    %Curl:3 = call @bwdCurl(%Tor, %Pol) : (!complex, !complex) -> (!real, !real, !real)
    %TPhysNl = call @nlScalar(%Vel#0, %Vel#1, %Vel#2, %TGrad#0, %TGrad#1, %TGrad#2) : (!real, !real, !real, !real, !real, !real) -> !real
    %VelNl:3 = call @nlVector(%Vel#0, %Vel#1, %Vel#2, %Curl#0, %Curl#1, %Curl#2, %TPhys) : (!real, !real, !real, !real, !real, !real, !real) -> (!real, !real, !real)
    %TNl = call @fwdScalar(%TPhysNl) : (!real) -> !complex
    %TorNl, %PolNl = call @fwdVector(%VelNl#0, %VelNl#1, %VelNl#2) : (!real, !real, !real) -> (!complex, !complex)
    return %TNl, %TorNl, %PolNl, %Vel#0, %Vel#1, %Vel#2: !complex, !complex, !complex, !real, !real, !real
}
   )mlir";

   MHDFloat Ra = spSim->eqParams()->nd(NonDimensional::Rayleigh::id());
   MHDFloat Pr = spSim->eqParams()->nd(NonDimensional::Prandtl::id());
   Graph::PhysicalParameters<MHDFloat> physParams;
   physParams.transport = 1.0;
   physParams.inertia = 1.0;
   physParams.buoyancy = Ra / Pr;
   spSim->addGraph(graphStr, physParams);
   #endif
}

void ITCModel::addStates(SharedStateGenerator spGen)
{
   // Shared pointer to equation
   Equations::SharedSphereExactScalarState spScalar;
   Equations::SharedSphereExactVectorState spVector;

   Spectral::Kernel::Complex3DMapType tSH;
   std::pair<Spectral::Kernel::Complex3DMapType::iterator, bool> ptSH;

   // Add temperature initial state generator
   spScalar =
      spGen->addEquation<Equations::SphereExactScalarState>(this->spBackend());
   spScalar->setIdentity(PhysicalNames::Temperature::id());
   switch (3)
   {
   case 0: {
      spScalar->setPhysicalNoise(1e-15);
   }
   break;

   case 1: {
      spScalar->setPhysicalConstant(1.0);
   }
   break;

   case 2: {
      tSH.clear();
      ptSH = tSH.insert(
         std::make_pair(std::make_pair(3, 3), std::map<int, MHDComplex>()));
      ptSH.first->second.insert(std::make_pair(7, MHDComplex(1.0, 2.0)));
      spScalar->setSpectralModes(tSH);
   }
   break;

   case 3: {
      auto spKernel = std::make_shared<Spectral::Kernel::MakeRandom>(
         spGen->ss().has(SpatialScheme::Feature::ComplexSpectrum));
      std::vector<MHDFloat> ratios = {1e4, 1e4, 1e4};
      spKernel->setRatio(ratios);
      spKernel->init(-1e-2, 1e-2);
      spScalar->setSrcKernel(spKernel);
   }
   break;
   }

   // Add velocity initial state generator
   spVector =
      spGen->addEquation<Equations::SphereExactVectorState>(this->spBackend());
   spVector->setIdentity(PhysicalNames::Velocity::id());
   switch (3)
   {
   // Toroidal only
   case 0: {
      // Toroidal
      tSH.clear();
      ptSH = tSH.insert(
         std::make_pair(std::make_pair(1, 1), std::map<int, MHDComplex>()));
      ptSH.first->second.insert(std::make_pair(7, MHDComplex(1.0)));
      spVector->setSpectralModes(FieldComponents::Spectral::TOR, tSH);
      // Poloidal
      tSH.clear();
      spVector->setSpectralModes(FieldComponents::Spectral::POL, tSH);
   }
   break;

   // Poloidal only
   case 1: {
      // Toroidal
      tSH.clear();
      spVector->setSpectralModes(FieldComponents::Spectral::TOR, tSH);
      // Poloidal
      tSH.clear();
      ptSH = tSH.insert(
         std::make_pair(std::make_pair(2, 0), std::map<int, MHDComplex>()));
      ptSH.first->second.insert(std::make_pair(7, MHDComplex(1.0)));
      spVector->setSpectralModes(FieldComponents::Spectral::POL, tSH);
   }
   break;

   // Toroidal & Poloidal
   case 2: {
      // Toroidal
      tSH.clear();
      ptSH = tSH.insert(
         std::make_pair(std::make_pair(1, 1), std::map<int, MHDComplex>()));
      ptSH.first->second.insert(std::make_pair(7, MHDComplex(1.0)));
      spVector->setSpectralModes(FieldComponents::Spectral::TOR, tSH);
      // Poloidal
      tSH.clear();
      ptSH = tSH.insert(
         std::make_pair(std::make_pair(2, 0), std::map<int, MHDComplex>()));
      ptSH.first->second.insert(std::make_pair(7, MHDComplex(1.0)));
      spVector->setSpectralModes(FieldComponents::Spectral::POL, tSH);
   }
   break;

   case 3: {
      auto spKernel = std::make_shared<Spectral::Kernel::MakeRandom>(
         spGen->ss().has(SpatialScheme::Feature::ComplexSpectrum));
      std::vector<MHDFloat> ratios = {1e4, 1e4, 1e4};
      spKernel->setRatio(ratios);
      spKernel->init(-1e-2, 1e-2);
      spVector->setSrcKernel(FieldComponents::Spectral::TOR, spKernel);
      spVector->setSrcKernel(FieldComponents::Spectral::POL, spKernel);
   }
   break;
   }

   // Add output file
   auto spOut =
      std::make_shared<Io::Variable::StateFileWriter>(spGen->ss().tag(),
         spGen->ss().has(SpatialScheme::Feature::RegularSpectrum));
   spOut->expect(PhysicalNames::Temperature::id());
   spOut->expect(PhysicalNames::Velocity::id());
   spGen->addHdf5OutputFile(spOut);
}

void ITCModel::addVisualizers(SharedVisualizationGenerator spVis)
{
   // Shared pointer to basic field visualizer
   Equations::SharedScalarFieldVisualizer spScalar;
   Equations::SharedVectorFieldVisualizer spVector;

   // Add temperature field visualization
   spScalar =
      spVis->addEquation<Equations::ScalarFieldVisualizer>(this->spBackend());
   spScalar->setFields(true, true);
   spScalar->setIdentity(PhysicalNames::Temperature::id());

   // Add velocity field visualization
   spVector =
      spVis->addEquation<Equations::VectorFieldVisualizer>(this->spBackend());
   spVector->setFields(true, false, true);
   spVector->setIdentity(PhysicalNames::Velocity::id());

   // Add output file
   auto spOut = std::make_shared<Io::Variable::VisualizationFileWriter>(
      spVis->ss().tag());
   spOut->expect(PhysicalNames::Temperature::id());
   spOut->expect(PhysicalNames::Velocity::id());
   spVis->addHdf5OutputFile(spOut);
}

std::map<std::string, std::map<std::string, int>> ITCModel::configTags() const
{
   std::map<std::string, int> onOff;
   onOff.emplace("enable", 1);

   std::map<std::string, int> options;
   options.emplace("enable", 0);
   options.emplace("numbered", 0);
   options.emplace("only_every", 1);

   std::map<std::string, std::map<std::string, int>> tags;

   // kinetic tags
   tags.emplace("kinetic_energy", onOff);
   tags.emplace("kinetic_l_spectrum", options);
   tags.emplace("kinetic_m_spectrum", options);
   tags.emplace("kinetic_n_spectrum", options);
   tags.emplace("kinetic_mode_spectrum", options);
   tags.emplace("kinetic_enstrophy", onOff);
   tags.emplace("kinetic_enstrophy_l_spectrum", options);
   tags.emplace("kinetic_enstrophy_m_spectrum", options);

   // temperature tags
   tags.emplace("temperature_energy", onOff);
   tags.emplace("temperature_l_spectrum", options);
   tags.emplace("temperature_m_spectrum", options);
   tags.emplace("temperature_n_spectrum", options);

   // diagnostic tags
   tags.emplace("angular_momentum", onOff);
   tags.emplace("nusselt", onOff);
   tags.emplace("temperature_mean", onOff);
   tags.emplace("velocity_max", onOff);

   return tags;
}

void ITCModel::addAsciiOutputFiles(SharedSimulation spSim)
{
   // Create Nusselt writer
   this->enableAsciiFile<Io::Variable::SphereNusseltWriter>("nusselt", "",
      PhysicalNames::Temperature::id(), spSim);

   // Create mean Temperature writer
   this->enableAsciiFile<Io::Variable::SphereScalarMeanWriter>(
      "temperature_mean", "temperature", PhysicalNames::Temperature::id(),
      spSim);

   // Create temperature energy writer
   this->enableAsciiFile<Io::Variable::SphereScalarEnergyWriter>(
      "temperature_energy", "temperature", PhysicalNames::Temperature::id(),
      spSim);

   // Create temperature L energy spectrum writer
   this->enableAsciiFile<Io::Variable::SphereScalarLSpectrumWriter>(
      "temperature_l_spectrum", "temperature", PhysicalNames::Temperature::id(),
      spSim);

   // Create temperature M energy spectrum writer
   this->enableAsciiFile<Io::Variable::SphereScalarMSpectrumWriter>(
      "temperature_m_spectrum", "temperature", PhysicalNames::Temperature::id(),
      spSim);

   // Create temperature N power spectrum writer
   this->enableAsciiFile<Io::Variable::SphereScalarNSpectrumWriter>(
      "temperature_n_spectrum", "temperature", PhysicalNames::Temperature::id(),
      spSim);

   // Create max absolute velocity writer
   this->enableAsciiFile<Io::Variable::SphereMaxAbsoluteFieldValueWriter>(
      "velocity_max", "velocity", PhysicalNames::Velocity::id(), spSim);

   // Create kinetic energy writer
   this->enableAsciiFile<Io::Variable::SphereTorPolEnergyWriter>(
      "kinetic_energy", "kinetic", PhysicalNames::Velocity::id(), spSim);

   // Create kinetic L energy spectrum writer
   this->enableAsciiFile<Io::Variable::SphereTorPolLSpectrumWriter>(
      "kinetic_l_spectrum", "kinetic", PhysicalNames::Velocity::id(), spSim);

   // Create kinetic M energy spectrum writer
   this->enableAsciiFile<Io::Variable::SphereTorPolMSpectrumWriter>(
      "kinetic_m_spectrum", "kinetic", PhysicalNames::Velocity::id(), spSim);

   // Create kinetic N power spectrum writer
   this->enableAsciiFile<Io::Variable::SphereTorPolNSpectrumWriter>(
      "kinetic_n_spectrum", "kinetic", PhysicalNames::Velocity::id(), spSim);

   // Create kinetic mode energy spectrum writer
   this->enableAsciiFile<Io::Variable::SphereTorPolModeSpectrumWriter>(
      "kinetic_mode_spectrum", "kinetic", PhysicalNames::Velocity::id(), spSim);

   // Create kinetic enstrophy writer
   this->enableAsciiFile<Io::Variable::SphereTorPolEnstrophyWriter>(
      "kinetic_enstrophy", "kinetic", PhysicalNames::Velocity::id(), spSim);

   // Create kinetic L enstrophy spectrum writer
   this->enableAsciiFile<Io::Variable::SphereTorPolEnstrophyLSpectrumWriter>(
      "kinetic_enstrophy_l_spectrum", "kinetic", PhysicalNames::Velocity::id(),
      spSim);

   // Create kinetic M enstrophy spectrum writer
   this->enableAsciiFile<Io::Variable::SphereTorPolEnstrophyMSpectrumWriter>(
      "kinetic_enstrophy_m_spectrum", "kinetic", PhysicalNames::Velocity::id(),
      spSim);

   // Create angular momentum writer
   this->enableAsciiFile<Io::Variable::SphereAngularMomentumWriter>(
      "angular_momentum", "", PhysicalNames::Velocity::id(), spSim);
}

} // namespace TC
} // namespace Sphere
} // namespace Boussinesq
} // namespace Model
} // namespace QuICC
