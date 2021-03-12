/**
 * @file ITCModel.cpp
 * @brief Source of the Boussinesq thermal convection in a sphere (Toroidal/Poloidal formulation)
 */

// Configuration includes
//

// System includes
//

// External includes
//

// Class include
//
#include "QuICC/Model/Boussinesq/Sphere/TC/ITCModel.hpp"

// Project includes
//
#include "QuICC/Model/Boussinesq/Sphere/TC/Transport.hpp"
#include "QuICC/Model/Boussinesq/Sphere/TC/Momentum.hpp"
#include "QuICC/Enums/FieldIds.hpp"
#include "QuICC/PhysicalNames/Temperature.hpp"
#include "QuICC/PhysicalNames/Velocity.hpp"
#include "QuICC/NonDimensional/Prandtl.hpp"
#include "QuICC/NonDimensional/Rayleigh.hpp"
#include "QuICC/Io/Variable/StateFileReader.hpp"
#include "QuICC/Io/Variable/StateFileWriter.hpp"
#include "QuICC/Io/Variable/VisualizationFileWriter.hpp"
#include "QuICC/Io/Variable/SphereNusseltWriter.hpp"
#include "QuICC/Io/Variable/SphereScalarEnergyWriter.hpp"
#include "QuICC/Io/Variable/SphereScalarLSpectrumWriter.hpp"
#include "QuICC/Io/Variable/SphereScalarMSpectrumWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolEnergyWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolLSpectrumWriter.hpp"
#include "QuICC/Io/Variable/SphereTorPolMSpectrumWriter.hpp"
#include "QuICC/Generator/States/RandomScalarState.hpp"
#include "QuICC/Generator/States/RandomVectorState.hpp"
#include "QuICC/Generator/States/SphereExactStateIds.hpp"
#include "QuICC/Generator/States/SphereExactScalarState.hpp"
#include "QuICC/Generator/States/SphereExactVectorState.hpp"
#include "QuICC/Generator/Visualizers/ScalarFieldVisualizer.hpp"
#include "QuICC/Generator/Visualizers/VectorFieldVisualizer.hpp"

namespace QuICC {

namespace Model {

namespace Boussinesq {

namespace Sphere {

namespace TC {

   VectorFormulation::Id ITCModel::SchemeFormulation()
   {
      return VectorFormulation::TORPOL;
   }

   void ITCModel::addEquations(SharedSimulation spSim)
   {
     // Add transport equation
      spSim->addEquation<Equations::Boussinesq::Sphere::TC::Transport>(this->spBackend());

      // Add Navier-Stokes equation
      spSim->addEquation<Equations::Boussinesq::Sphere::TC::Momentum>(this->spBackend());
   }

   void ITCModel::addStates(SharedStateGenerator spGen)
   {
      // Generate "exact" solutions (trigonometric or monomial)
      if(true)
      {
         // Shared pointer to equation
         Equations::SharedSphereExactScalarState spScalar;
         Equations::SharedSphereExactVectorState spVector;

         Equations::SHMapType tSH;
         std::pair<Equations::SHMapType::iterator,bool> ptSH;

         // Add temperature initial state generator
         spScalar = spGen->addEquation<Equations::SphereExactScalarState>();
         spScalar->setIdentity(PhysicalNames::Temperature::id());
         switch(0)
         {
            case 0:
               spScalar->setSpectralType(Equations::SphereExactStateIds::HARMONIC);
               tSH.clear();
               ptSH = tSH.insert(std::make_pair(std::make_pair(3,3), std::map<int,MHDComplex>()));
               ptSH.first->second.insert(std::make_pair(7, MHDComplex(1.0,2.0)));
               spScalar->setHarmonicOptions(tSH);
               break;
         }

         // Add velocity initial state generator
         spVector = spGen->addEquation<Equations::SphereExactVectorState>();
         spVector->setIdentity(PhysicalNames::Velocity::id());
         switch(2)
         {
            case 0:
               // Toroidal
               spVector->setSpectralType(Equations::SphereExactStateIds::HARMONIC);
               tSH.clear();
               ptSH = tSH.insert(std::make_pair(std::make_pair(1,1), std::map<int,MHDComplex>()));
               ptSH.first->second.insert(std::make_pair(7, MHDComplex(1.0)));
               spVector->setHarmonicOptions(FieldComponents::Spectral::TOR, tSH);
               break;

            case 1:
               // Poloidal
               spVector->setSpectralType(Equations::SphereExactStateIds::HARMONIC);
               tSH.clear();
               ptSH = tSH.insert(std::make_pair(std::make_pair(2,0), std::map<int,MHDComplex>()));
               ptSH.first->second.insert(std::make_pair(7, MHDComplex(1.0)));
               spVector->setHarmonicOptions(FieldComponents::Spectral::POL, tSH);
               break;

            case 2:
               // Toroidal
               spVector->setSpectralType(Equations::SphereExactStateIds::HARMONIC);
               tSH.clear();
               ptSH = tSH.insert(std::make_pair(std::make_pair(1,1), std::map<int,MHDComplex>()));
               ptSH.first->second.insert(std::make_pair(7, MHDComplex(1.0)));
               spVector->setHarmonicOptions(FieldComponents::Spectral::TOR, tSH);
               // Poloidal
               tSH.clear();
               ptSH = tSH.insert(std::make_pair(std::make_pair(2,0), std::map<int,MHDComplex>()));
               ptSH.first->second.insert(std::make_pair(7, MHDComplex(1.0)));
               spVector->setHarmonicOptions(FieldComponents::Spectral::POL, tSH);
               break;
         }

      // Generate random spectrum
      } else
      {
         // Shared pointer to random initial state equation
         Equations::SharedRandomScalarState spScalar;
         Equations::SharedRandomVectorState spVector;

         // Add scalar random initial state generator
         spVector = spGen->addEquation<Equations::RandomVectorState>();
         spVector->setIdentity(PhysicalNames::Velocity::id());
         spVector->setSpectrum(FieldComponents::Spectral::TOR, -1e-2, 1e-2, 1e4, 1e4, 1e4);
         spVector->setSpectrum(FieldComponents::Spectral::POL, -1e-2, 1e-2, 1e4, 1e4, 1e4);

         // Add scalar random initial state generator
         spScalar = spGen->addEquation<Equations::RandomScalarState>();
         spScalar->setIdentity(PhysicalNames::Temperature::id());
         spScalar->setSpectrum(-1e-2, 1e-2, 1e4, 1e4, 1e4);
      }

      // Add output file
      auto spOut = std::make_shared<Io::Variable::StateFileWriter>(spGen->ss().tag(), spGen->ss().has(SpatialScheme::Feature::RegularSpectrum));
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
      spScalar = spVis->addEquation<Equations::ScalarFieldVisualizer>();
      spScalar->setFields(true, true);
      spScalar->setIdentity(PhysicalNames::Temperature::id());

      // Add velocity field visualization
      spVector = spVis->addEquation<Equations::VectorFieldVisualizer>();
      spVector->setFields(true, false, true);
      spVector->setIdentity(PhysicalNames::Velocity::id());

      // Add output file
      auto spOut = std::make_shared<Io::Variable::VisualizationFileWriter>(spVis->ss().tag());
      spOut->expect(PhysicalNames::Temperature::id());
      spOut->expect(PhysicalNames::Velocity::id());
      spVis->addHdf5OutputFile(spOut);
   }

   void ITCModel::addAsciiOutputFiles(SharedSimulation spSim)
   {
      // Create Nusselt writer
      auto spNusselt = std::make_shared<Io::Variable::SphereNusseltWriter>("", spSim->ss().tag());
      spNusselt->expect(PhysicalNames::Temperature::id());
      spSim->addAsciiOutputFile(spNusselt);

      // Create temperature energy writer
      auto spTemp = std::make_shared<Io::Variable::SphereScalarEnergyWriter>("temperature", spSim->ss().tag());
      spTemp->expect(PhysicalNames::Temperature::id());
      spSim->addAsciiOutputFile(spTemp);

#if 0
      // Create temperature L energy spectrum writer
      auto spTempL = std::make_shared<Io::Variable::SphereScalarLSpectrumWriter>("temperature", spSim->ss().tag());
      spTempL->expect(PhysicalNames::Temperature::id());
      //spTempL->numberOutput();
      spSim->addAsciiOutputFile(spTempL);

      // Create temperature M energy spectrum writer
      auto spTempM = std::make_shared<Io::Variable::SphereScalarMSpectrumWriter>("temperature", spSim->ss().tag());
      spTempM->expect(PhysicalNames::Temperature::id());
      //spTempM->numberOutput();
      spSim->addAsciiOutputFile(spTempM);
#endif

      // Create kinetic energy writer
      auto spKinetic = std::make_shared<Io::Variable::SphereTorPolEnergyWriter>("kinetic", spSim->ss().tag());
      spKinetic->expect(PhysicalNames::Velocity::id());
      spSim->addAsciiOutputFile(spKinetic);

#if 0
      // Create kinetic L energy spectrum writer
      auto spKineticL = std::make_shared<Io::Variable::SphereTorPolLSpectrumWriter>("kinetic", spSim->ss().tag());
      spKineticL->expect(PhysicalNames::Velocity::id());
      //spKineticL->numberOutput();
      spSim->addAsciiOutputFile(spKineticL);

      // Create kinetic M energy spectrum writer
      auto spKineticM = std::make_shared<Io::Variable::SphereTorPolMSpectrumWriter>("kinetic", spSim->ss().tag());
      spKineticM->expect(PhysicalNames::Velocity::id());
      //spKineticM->numberOutput();
      spSim->addAsciiOutputFile(spKineticM);
#endif
   }

}
}
}
}
}
