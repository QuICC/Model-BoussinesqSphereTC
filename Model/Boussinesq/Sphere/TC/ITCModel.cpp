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
