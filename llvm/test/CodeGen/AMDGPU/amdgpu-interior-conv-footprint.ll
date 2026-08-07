; REQUIRES: asserts
; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes='loop-simplify,lcssa,amdgpu-interior-tile-split' \
; RUN:   -debug-only=amdgpu-interior-tile-split -disable-output %s 2>&1 | FileCheck %s --check-prefix=DBG
; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes='loop-simplify,lcssa,amdgpu-interior-tile-split' \
; RUN:   -S %s | FileCheck %s --check-prefix=CFG

; Conv-style cooperative staging: CTA-uniform halo bases (possibly negative
; after pad) plus per-element H/W guards before a shared barrier.  The pass
; inserts a uniform footprint selector, clones staging, and strips proven
; bounds branches on the interior path.

declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @llvm.amdgcn.workgroup.id.y()
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()

; DBG: Conv footprint recovered {{[2-9]}} dims from {{[2-9]}} bases
; DBG: Cloned conv footprint interior staging in conv_footprint_staging

define amdgpu_kernel void @conv_footprint_staging(ptr addrspace(1) %in,
                                                  ptr addrspace(3) %sIn,
                                                  i32 %H, i32 %W) {
entry:
  %wg.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %wg.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %y.tile = mul i32 %wg.y, 32
  %x.tile = mul i32 %wg.x, 64
  %y.base = sub i32 %y.tile, 2
  %x.base = sub i32 %x.tile, 2
  br label %staging

staging:
  %iy = and i32 %lane, 31
  ; Fanl conv11 W shape: ix4*4 narrowed through i16 then zexted.
  %t1 = udiv i32 %lane, 42
  %ix4 = urem i32 %t1, 18
  %x.off32 = shl nuw nsw i32 %ix4, 2
  %x.off16 = trunc i32 %x.off32 to i16
  %xoff = zext i16 %x.off16 to i32
  %y = add i32 %y.base, %iy
  %y.ok = icmp ult i32 %y, %H
  br i1 %y.ok, label %x.check, label %barrier

x.check:
  %x0 = add i32 %x.base, %xoff
  %x1 = add i32 %x0, 1
  %x2 = add i32 %x0, 2
  %x3 = add i32 %x0, 3
  %in0 = icmp ult i32 %x0, %W
  %in1 = icmp ult i32 %x1, %W
  %in2 = icmp ult i32 %x2, %W
  %in3 = icmp ult i32 %x3, %W
  %in01 = and i1 %in0, %in1
  %in23 = and i1 %in2, %in3
  %x.ok = and i1 %in01, %in23
  br i1 %x.ok, label %load, label %barrier

load:
  %y64 = zext i32 %y to i64
  %x64 = zext i32 %x0 to i64
  %ptr = getelementptr float, ptr addrspace(1) %in, i64 %y64
  %val = load float, ptr addrspace(1) %ptr, align 4
  %sptr = getelementptr float, ptr addrspace(3) %sIn, i32 %lane
  store float %val, ptr addrspace(3) %sptr, align 4
  br label %barrier

barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %compute

compute:
  ; Post-barrier work stays shared (not cloned).
  ret void
}

; CFG-LABEL: define amdgpu_kernel void @conv_footprint_staging(
; Selector may be SSA-numbered (%interior.conv.full or %interior.conv.fullN).
; CFG: br i1 %{{interior.conv.full[0-9]*}}, label %staging.interior, label %staging
; CFG-LABEL: staging:
; CFG: br i1 %y.ok, label %x.check, label %barrier
; CFG-LABEL: x.check:
; CFG: br i1 %x.ok, label %load, label %barrier
; CFG-LABEL: barrier:
; CFG-COUNT-1: call void @llvm.amdgcn.s.barrier()
; CFG: br label %compute
; CFG-LABEL: compute:
; CFG: ret void
; CFG-LABEL: staging.interior:
; Proven H/W guards are folded away on the interior clone.
; CFG: br label %x.check.interior
; CFG-LABEL: x.check.interior:
; CFG: br label %load.interior
; CFG-LABEL: load.interior:
; CFG: br label %barrier
; CFG-NOT: barrier.interior
; CFG-NOT: compute.interior

define amdgpu_kernel void @conv_footprint_urem_offsets(ptr addrspace(1) %in,
                                                       ptr addrspace(3) %sIn,
                                                       i32 %H, i32 %W) {
entry:
  %wg.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %wg.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %lane = call i32 @llvm.amdgcn.workitem.id.x()
  %y.tile = mul i32 %wg.y, 32
  %x.tile = mul i32 %wg.x, 64
  %y.base = sub i32 %y.tile, 2
  %x.base = sub i32 %x.tile, 2
  br label %loop

loop:
  %vid = phi i32 [ %lane, %entry ], [ %vid.next, %latch ]
  %iy = urem i32 %vid, 42
  %t1 = udiv i32 %vid, 42
  %ix4 = urem i32 %t1, 18
  %x.off = mul i32 %ix4, 4
  %y = add i32 %y.base, %iy
  %y.ok = icmp ult i32 %y, %H
  br i1 %y.ok, label %x.check, label %latch

x.check:
  %x0 = add i32 %x.base, %x.off
  %x1.off = add i32 %x.off, 1
  %x1 = add i32 %x.base, %x1.off
  %x2.off = add i32 %x.off, 2
  %x2 = add i32 %x.base, %x2.off
  %x3.off = add i32 %x.off, 3
  %x3 = add i32 %x.base, %x3.off
  %in0 = icmp ult i32 %x0, %W
  %in1 = icmp ult i32 %x1, %W
  %in2 = icmp ult i32 %x2, %W
  %in3 = icmp ult i32 %x3, %W
  %in01 = and i1 %in0, %in1
  %in23 = and i1 %in2, %in3
  %x.ok = and i1 %in01, %in23
  br i1 %x.ok, label %load, label %latch

load:
  %ptr = getelementptr float, ptr addrspace(1) %in, i32 %x0
  %val = load float, ptr addrspace(1) %ptr, align 4
  %sptr = getelementptr float, ptr addrspace(3) %sIn, i32 %lane
  store float %val, ptr addrspace(3) %sptr, align 4
  br label %latch

latch:
  %vid.next = add i32 %vid, 128
  %more = icmp ult i32 %vid.next, 2048
  br i1 %more, label %loop, label %barrier

barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %compute2

compute2:
  ret void
}

; DBG: Cloned conv footprint interior staging in conv_footprint_urem_offsets
; CFG-LABEL: define amdgpu_kernel void @conv_footprint_urem_offsets(
; CFG: br i1 %{{interior.conv.full[0-9]*}}, label %loop.interior, label %loop
; CFG-LABEL: loop.interior:
; CFG: br label %x.check.interior
; CFG-LABEL: x.check.interior:
; CFG: br label %load.interior
; CFG-NOT: barrier.interior
