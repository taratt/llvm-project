//===-- AMDGPUInteriorTileSplit.cpp - Find tiled boundary checks ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Identifies divergent tile-boundary checks that combine workgroup and
/// workitem IDs. A later transformation will use these candidates to create a
/// workgroup-uniform interior path and retain the original guarded edge path.
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "amdgpu-interior-tile-split"

using namespace llvm;

STATISTIC(NumInteriorTileCandidates,
          "Number of divergent tile-boundary checks found");

namespace {

enum IDDependency : unsigned {
  DependsOnNone = 0,
  DependsOnWorkgroupID = 1u << 0,
  DependsOnWorkitemID = 1u << 1,
};

static unsigned getIDDependencies(Value *V,
                                  SmallPtrSetImpl<Value *> &Visited) {
  if (!Visited.insert(V).second)
    return DependsOnNone;

  if (auto *II = dyn_cast<IntrinsicInst>(V)) {
    StringRef Name = II->getCalledFunction()->getName();
    if (Name.starts_with("llvm.amdgcn.workgroup.id."))
      return DependsOnWorkgroupID;
    if (Name.starts_with("llvm.amdgcn.workitem.id."))
      return DependsOnWorkitemID;
  }

  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return DependsOnNone;

  unsigned Dependencies = DependsOnNone;
  for (Value *Operand : I->operands())
    Dependencies |= getIDDependencies(Operand, Visited);
  return Dependencies;
}

static bool findInteriorTileCandidates(Function &F, UniformityInfo &UI) {
  if (!AMDGPU::isEntryFunctionCC(F.getCallingConv()))
    return false;

  for (BasicBlock &BB : F) {
    auto *Branch = dyn_cast<CondBrInst>(BB.getTerminator());
    if (!Branch || UI.isUniformTerminator(Branch))
      continue;

    SmallPtrSet<Value *, 16> Visited;
    const unsigned Dependencies =
        getIDDependencies(Branch->getCondition(), Visited);
    if ((Dependencies & (DependsOnWorkgroupID | DependsOnWorkitemID)) !=
        (DependsOnWorkgroupID | DependsOnWorkitemID))
      continue;

    ++NumInteriorTileCandidates;
    LLVM_DEBUG(dbgs() << "Potential interior-tile boundary check in "
                      << F.getName() << ":\n"
                      << *Branch << '\n');
  }

  // This first version only identifies candidates. It intentionally leaves
  // CFG and IR unchanged until the full-tile proof and cloning transform are
  // implemented.
  return false;
}

class AMDGPUInteriorTileSplitLegacy : public FunctionPass {
public:
  static char ID;

  AMDGPUInteriorTileSplitLegacy() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AMDGPU Interior Tile Split";
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    UniformityInfo &UI =
        getAnalysis<UniformityInfoWrapperPass>().getUniformityInfo();
    return findInteriorTileCandidates(F, UI);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<UniformityInfoWrapperPass>();
    AU.setPreservesAll();
  }
};

} // end anonymous namespace

PreservedAnalyses
AMDGPUInteriorTileSplitPass::run(Function &F, FunctionAnalysisManager &FAM) {
  UniformityInfo &UI = FAM.getResult<UniformityInfoAnalysis>(F);
  findInteriorTileCandidates(F, UI);
  return PreservedAnalyses::all();
}

INITIALIZE_PASS_BEGIN(AMDGPUInteriorTileSplitLegacy, DEBUG_TYPE,
                      "Find AMDGPU interior tile split candidates", false,
                      true)
INITIALIZE_PASS_DEPENDENCY(UniformityInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUInteriorTileSplitLegacy, DEBUG_TYPE,
                    "Find AMDGPU interior tile split candidates", false, true)

char AMDGPUInteriorTileSplitLegacy::ID = 0;

FunctionPass *llvm::createAMDGPUInteriorTileSplitLegacy() {
  return new AMDGPUInteriorTileSplitLegacy();
}
