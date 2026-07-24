; REQUIRES: asserts
; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes='loop-simplify,lcssa,amdgpu-interior-tile-split' \
; RUN:   -debug-only=amdgpu-interior-tile-split -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes='loop-simplify,lcssa,amdgpu-interior-tile-split' \
; RUN:   -verify-each -S %s | FileCheck %s --check-prefix=CFG

; The broad candidate diagnostic is retained for non-canonical boundary
; checks.  The second kernel has a closed staging region, which is cloned
; behind the uniform full-tile dispatch while its barrier remains shared.

declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @candidate(ptr addrspace(1) %out, i32 %bound) {
entry:
  %workgroup = call i32 @llvm.amdgcn.workgroup.id.x()
  %workitem = call i32 @llvm.amdgcn.workitem.id.x()
  %tile-base = mul i32 %workgroup, 64
  %index = add i32 %tile-base, %workitem
  %in-bounds = icmp ult i32 %index, %bound
  br i1 %in-bounds, label %store, label %exit

store:
  %ptr = getelementptr float, ptr addrspace(1) %out, i32 %index
  store float 0.000000e+00, ptr addrspace(1) %ptr, align 4
  br label %exit

exit:
  ret void
}

declare i32 @llvm.amdgcn.workgroup.id.y()
declare void @llvm.amdgcn.s.barrier()
declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.umin.i32(i32, i32)

define amdgpu_kernel void @canonical_staging(ptr addrspace(1) %out, i32 %m,
                                             i32 %n, i32 %k) {
entry:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %workitem = call i32 @llvm.amdgcn.workitem.id.x()
  %x.base = shl i32 %workgroup.x, 7
  %y.base = shl i32 %workgroup.y, 7
  %x.last = add i32 %x.base, 127
  %y.last = add i32 %y.base, 127
  %n.remaining = sub i32 %n, %x.base
  %n.nonnegative = call i32 @llvm.smax.i32(i32 %n.remaining, i32 0)
  %n.extent = call i32 @llvm.smin.i32(i32 %n.nonnegative, i32 128)
  %m.remaining = sub i32 %m, %y.base
  %m.nonnegative = call i32 @llvm.smax.i32(i32 %m.remaining, i32 0)
  %m.extent = call i32 @llvm.smin.i32(i32 %m.nonnegative, i32 128)
  %k.remaining = sub i32 %k, 0
  %k.extent = call i32 @llvm.smin.i32(i32 %k.remaining, i32 32)
  %full.n = icmp ult i32 %x.last, %n
  %full.m = icmp ult i32 %y.last, %m
  %full.k = icmp uge i32 %k, 32
  %out.int = ptrtoint ptr addrspace(1) %out to i64
  %out.mask = and i64 %out.int, 15
  %out.aligned = icmp eq i64 %out.mask, 0
  %full.mn = and i1 %full.m, %full.n
  %full.mnk = and i1 %full.mn, %full.k
  %full = and i1 %full.mnk, %out.aligned
  br i1 %full, label %staging, label %edge

staging:
  %lane = and i32 %workitem, 127
  %index = add i32 %x.base, %lane
  %in.bounds = icmp ult i32 %index, %n
  br i1 %in.bounds, label %stage.store, label %stage.skip

stage.store:
  %ptr = getelementptr float, ptr addrspace(1) %out, i32 %index
  store float 0.000000e+00, ptr addrspace(1) %ptr, align 4
  br label %barrier

stage.skip:
  br label %barrier

barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %exit

edge:
  br label %exit

exit:
  ret void
}

; The dynamic K loop is already in LoopSimplify and LCSSA form.  Its preheader
; has a canonical, CTA-uniform M/N/alignment proof.  The pass must run a
; 32-element prefix and enter the original guarded tail only for K % 32.
define amdgpu_kernel void @canonical_k_loop(ptr addrspace(1) %out, i32 %m,
                                            i32 %n, i32 %k) {
entry:
  br label %dispatch

dispatch:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %x.base = shl i32 %workgroup.x, 7
  %y.base = shl i32 %workgroup.y, 7
  %x.last = add i32 %x.base, 127
  %y.last = add i32 %y.base, 127
  %n.remaining = sub i32 %n, %x.base
  %n.nonnegative = call i32 @llvm.smax.i32(i32 %n.remaining, i32 0)
  %n.extent = call i32 @llvm.smin.i32(i32 %n.nonnegative, i32 128)
  %m.remaining = sub i32 %m, %y.base
  %m.nonnegative = call i32 @llvm.smax.i32(i32 %m.remaining, i32 0)
  %m.extent = call i32 @llvm.smin.i32(i32 %m.nonnegative, i32 128)
  %k.remaining = sub i32 %k, 0
  %k.extent = call i32 @llvm.smin.i32(i32 %k.remaining, i32 32)
  %full.n = icmp ult i32 %x.last, %n
  %full.m = icmp ult i32 %y.last, %m
  %full.k = icmp uge i32 %k, 32
  %out.int = ptrtoint ptr addrspace(1) %out to i64
  %out.mask = and i64 %out.int, 15
  %out.aligned = icmp eq i64 %out.mask, 0
  %full.mn = and i1 %full.m, %full.n
  %full.mnk = and i1 %full.mn, %full.k
  %full = and i1 %full.mnk, %out.aligned
  br label %k.preheader

k.preheader:
  br label %k.header

k.header:
  %i = phi i32 [ 0, %k.preheader ], [ %next, %k.latch ]
  %acc = phi i32 [ 0, %k.preheader ], [ %acc.next, %k.latch ]
  br label %k.body

k.body:
  %k.remaining.loop = sub i32 %k, %i
  %k.extent.loop = call i32 @llvm.umin.i32(i32 %k.remaining.loop, i32 32)
  %k.full.loop = icmp uge i32 %k.extent.loop, 32
  br i1 %k.full.loop, label %k.barrier, label %k.skip

k.barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %k.latch

k.skip:
  br label %k.latch

k.latch:
  %next = add nuw i32 %i, 32
  %acc.next = add i32 %acc, 1
  %more = icmp ult i32 %next, %k
  br i1 %more, label %k.header, label %k.exit

; A dynamic bound is essential: a fixed trip count cannot exercise both the
; full-prefix and guarded-tail paths.
k.exit:
  %result.lcssa = phi i32 [ %next, %k.latch ]
  %acc.lcssa = phi i32 [ %acc.next, %k.latch ]
  store i32 %result.lcssa, ptr addrspace(1) %out, align 4
  store i32 %acc.lcssa, ptr addrspace(1) %out, align 4
  ret void
}

; This multi-tile K loop has its canonical M/N clamp remainders in the actual
; preheader.  The pass synthesizes its static selector there, then removes the
; exact M/N lane checks and K tile check from the cloned full-K prefix.
define amdgpu_kernel void @canonical_synthesized_k_loop(
    ptr addrspace(1) %out, i32 %m, i32 %n, i32 %k) {
entry:
  br label %dispatch

dispatch:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %workitem = call i32 @llvm.amdgcn.workitem.id.x()
  %x.base = shl i32 %workgroup.x, 7
  %y.base = shl i32 %workgroup.y, 7
  br label %k.preheader

k.preheader:
  %n.remaining = sub i32 %n, %x.base
  %n.nonnegative = call i32 @llvm.smax.i32(i32 %n.remaining, i32 0)
  %n.extent = call i32 @llvm.smin.i32(i32 %n.nonnegative, i32 128)
  %m.remaining = sub i32 %m, %y.base
  %m.nonnegative = call i32 @llvm.smax.i32(i32 %m.remaining, i32 0)
  %m.extent = call i32 @llvm.smin.i32(i32 %m.nonnegative, i32 128)
  %k.remaining = sub i32 %k, 0
  %k.extent = call i32 @llvm.smin.i32(i32 %k.remaining, i32 32)
  %out.int = ptrtoint ptr addrspace(1) %out to i64
  %out.mask = and i64 %out.int, 15
  %out.aligned = icmp eq i64 %out.mask, 0
  br label %k.header

k.header:
  %i = phi i32 [ 0, %k.preheader ], [ %next, %k.latch ]
  br label %k.body

k.body:
  ; The lane expression is deliberately affine but not the old exact
  ; workitem-id spelling.  Its largest value is 63, below the 128-wide tile.
  %lane.masked = and i32 %workitem, 31
  %lane.narrow = trunc i32 %lane.masked to i8
  %lane.extended = zext i8 %lane.narrow to i32
  %lane.scaled = mul i32 %lane.extended, 2
  %lane = add i32 %lane.scaled, 1
  %m.index = add i32 %y.base, %lane
  %m.in.bounds = icmp slt i32 %m.index, %m
  br i1 %m.in.bounds, label %k.n.check, label %k.barrier

k.n.check:
  %n.index = add i32 %x.base, %lane
  %n.in.bounds = icmp slt i32 %n.index, %n
  br i1 %n.in.bounds, label %k.k.check, label %k.barrier

k.k.check:
  %k.remaining.loop = sub i32 %k, %i
  %k.extent.loop = call i32 @llvm.smin.i32(i32 %k.remaining.loop, i32 32)
  %k.full.loop = icmp sge i32 %k.extent.loop, 32
  br i1 %k.full.loop, label %k.barrier, label %k.skip

k.barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %k.latch

k.skip:
  br label %k.latch

k.latch:
  %next = add nuw i32 %i, 32
  %more = icmp ult i32 %next, %k
  br i1 %more, label %k.header, label %k.exit

k.exit:
  %result.lcssa = phi i32 [ %next, %k.latch ]
  store i32 %result.lcssa, ptr addrspace(1) %out, align 4
  ret void
}

; Clang can lower the M/N clamp setup in the block immediately before the
; otherwise-empty K preheader.  That predecessor dominates the preheader, so
; its values are safe selector inputs without looking through arbitrary CFG.
define amdgpu_kernel void @canonical_predecessor_setup_k_loop(
    ptr addrspace(1) %out, i32 %m, i32 %n, i32 %k) {
entry:
  br label %dispatch

dispatch:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %workitem = call i32 @llvm.amdgcn.workitem.id.x()
  %x.base = shl i32 %workgroup.x, 7
  %y.base = shl i32 %workgroup.y, 7
  %n.remaining = sub i32 %n, %x.base
  %n.nonnegative = call i32 @llvm.smax.i32(i32 %n.remaining, i32 0)
  %n.extent = call i32 @llvm.smin.i32(i32 %n.nonnegative, i32 128)
  %m.remaining = sub i32 %m, %y.base
  %m.nonnegative = call i32 @llvm.smax.i32(i32 %m.remaining, i32 0)
  %m.extent = call i32 @llvm.smin.i32(i32 %m.nonnegative, i32 128)
  %k.remaining = sub i32 %k, 0
  %k.extent = call i32 @llvm.smin.i32(i32 %k.remaining, i32 32)
  %out.int = ptrtoint ptr addrspace(1) %out to i64
  %out.mask = and i64 %out.int, 15
  %out.aligned = icmp eq i64 %out.mask, 0
  br label %k.preheader

k.preheader:
  br label %k.header

k.header:
  %i = phi i32 [ 0, %k.preheader ], [ %next, %k.latch ]
  br label %k.body

k.body:
  %lane = and i32 %workitem, 127
  %m.index = add i32 %y.base, %lane
  %m.in.bounds = icmp slt i32 %m.index, %m
  br i1 %m.in.bounds, label %k.n.check, label %k.barrier

k.n.check:
  %n.index = add i32 %x.base, %lane
  %n.in.bounds = icmp slt i32 %n.index, %n
  br i1 %n.in.bounds, label %k.k.check, label %k.barrier

k.k.check:
  %k.remaining.loop = sub i32 %k, %i
  %k.extent.loop = call i32 @llvm.smin.i32(i32 %k.remaining.loop, i32 32)
  %k.full.loop = icmp sge i32 %k.extent.loop, 32
  br i1 %k.full.loop, label %k.barrier, label %k.skip

k.barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %k.latch

k.skip:
  br label %k.latch

k.latch:
  %next = add nuw i32 %i, 32
  %more = icmp ult i32 %next, %k
  br i1 %more, label %k.header, label %k.exit

k.exit:
  %result.lcssa = phi i32 [ %next, %k.latch ]
  store i32 %result.lcssa, ptr addrspace(1) %out, align 4
  ret void
}

; The staging loop is nested inside the 32-wide K loop.  Its offset reaches
; the M/N/K guards through affine adds, as in real GEMM IR.  The combined
; select is the short-circuit lowering of K-in-bounds && N-in-bounds.  Only
; the clone's true-path select guard is removed; the original edge stays
; guarded.
define amdgpu_kernel void @canonical_nested_staging_k_loop(
    ptr addrspace(1) %out, i32 %m, i32 %n, i32 %k) {
entry:
  br label %dispatch

dispatch:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %x.base = shl i32 %workgroup.x, 7
  %y.base = shl i32 %workgroup.y, 7
  br label %k.preheader

k.preheader:
  %n.remaining = sub i32 %n, %x.base
  %n.nonnegative = call i32 @llvm.smax.i32(i32 %n.remaining, i32 0)
  %n.extent = call i32 @llvm.smin.i32(i32 %n.nonnegative, i32 128)
  %m.remaining = sub i32 %m, %y.base
  %m.nonnegative = call i32 @llvm.smax.i32(i32 %m.remaining, i32 0)
  %m.extent = call i32 @llvm.smin.i32(i32 %m.nonnegative, i32 128)
  %k.remaining = sub i32 %k, 0
  %k.extent = call i32 @llvm.smin.i32(i32 %k.remaining, i32 32)
  %out.int = ptrtoint ptr addrspace(1) %out to i64
  %out.mask = and i64 %out.int, 15
  %out.aligned = icmp eq i64 %out.mask, 0
  br label %k.header

k.header:
  %i = phi i32 [ 0, %k.preheader ], [ %next, %k.latch ]
  br label %k.body

k.body:
  br label %stage.header

stage.header:
  %stage.i = phi i32 [ 0, %k.body ], [ %stage.next, %stage.latch ]
  br label %stage.m.check

stage.m.check:
  %m.index = add i32 %y.base, %stage.i
  %m.in.bounds = icmp slt i32 %m.index, %m
  br i1 %m.in.bounds, label %stage.n.check, label %stage.latch

stage.n.check:
  %n.index = add i32 %x.base, %stage.i
  %n.in.bounds = icmp slt i32 %n.index, %n
  %k.index = add i32 %i, %stage.i
  %k.in.bounds = icmp slt i32 %k.index, %k
  %or.cond = select i1 %k.in.bounds, i1 %n.in.bounds, i1 false
  br i1 %or.cond, label %stage.k.check, label %stage.latch

stage.k.check:
  br label %stage.barrier

stage.barrier:
  call void @llvm.amdgcn.s.barrier()
  br label %stage.latch

stage.latch:
  %stage.next = add nuw i32 %stage.i, 1
  %stage.more = icmp ult i32 %stage.next, 32
  br i1 %stage.more, label %stage.header, label %k.latch

k.latch:
  %next = add nuw i32 %i, 32
  %more = icmp ult i32 %next, %k
  br i1 %more, label %k.header, label %k.exit

k.exit:
  %result.lcssa = phi i32 [ %next, %k.latch ]
  store i32 %result.lcssa, ptr addrspace(1) %out, align 4
  ret void
}

; The equivalent one-K-tile loop is structurally proven to execute once.
; Synthesis includes all M/N/K/alignment predicates before it versions the
; loop.
define amdgpu_kernel void @canonical_synthesized_one_k_loop(
    ptr addrspace(1) %out, i32 %m, i32 %n, i32 %k) {
entry:
  br label %dispatch

dispatch:
  %workgroup.x = call i32 @llvm.amdgcn.workgroup.id.x()
  %workgroup.y = call i32 @llvm.amdgcn.workgroup.id.y()
  %x.base = shl i32 %workgroup.x, 7
  %y.base = shl i32 %workgroup.y, 7
  %n.remaining = sub i32 %n, %x.base
  %n.nonnegative = call i32 @llvm.smax.i32(i32 %n.remaining, i32 0)
  %n.extent = call i32 @llvm.smin.i32(i32 %n.nonnegative, i32 128)
  %m.remaining = sub i32 %m, %y.base
  %m.nonnegative = call i32 @llvm.smax.i32(i32 %m.remaining, i32 0)
  %m.extent = call i32 @llvm.smin.i32(i32 %m.nonnegative, i32 128)
  %k.remaining = sub i32 %k, 0
  %k.extent = call i32 @llvm.smin.i32(i32 %k.remaining, i32 32)
  %out.int = ptrtoint ptr addrspace(1) %out to i64
  %out.mask = and i64 %out.int, 15
  %out.aligned = icmp eq i64 %out.mask, 0
  br label %k.preheader

k.preheader:
  br label %k.header

k.header:
  %i = phi i32 [ 0, %k.preheader ], [ %next, %k.latch ]
  br label %k.body

k.body:
  call void @llvm.amdgcn.s.barrier()
  %next = add nuw i32 %i, 1
  %done = icmp eq i32 %next, 1
  br i1 %done, label %k.exit, label %k.latch

k.latch:
  br label %k.header

k.exit:
  %result.lcssa = phi i32 [ %next, %k.body ]
  store i32 %result.lcssa, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK: Cloned canonical interior-tile staging region in canonical_staging at entry; removed 1 proven lane bounds branch(es); fast staging entry staging.interior, fallback staging, shared barrier barrier
; CHECK: Split canonical interior K loop in canonical_k_loop at dispatch; prefix loop k.header.interior, guarded tail k.header; removed 1 proven guard(s), shared live-out exit k.exit
; CHECK: Split canonical interior K loop in canonical_synthesized_k_loop at k.preheader; prefix loop k.header.interior, guarded tail k.header; removed 3 proven guard(s), shared live-out exit k.exit
; CHECK: Interior K-loop candidate k.header: preheader=k.preheader HasM=1 HasN=1 canSplit=1
; CHECK: Split canonical interior K loop in canonical_predecessor_setup_k_loop at k.preheader; prefix loop k.header.interior, guarded tail k.header; removed 3 proven guard(s), shared live-out exit k.exit
; CHECK: Split canonical interior K loop in canonical_nested_staging_k_loop at k.preheader; prefix loop k.header.interior, guarded tail k.header; removed 2 proven guard(s), shared live-out exit k.exit
; CHECK: Potential interior-tile boundary check in candidate:

; CFG-LABEL: define amdgpu_kernel void @canonical_staging(
; CFG: br i1 %full, label %staging.interior, label %edge
; CFG-LABEL: staging:
; CFG: br i1 %in.bounds, label %stage.store, label %stage.skip
; CFG-LABEL: barrier:
; CFG-COUNT-1: call void @llvm.amdgcn.s.barrier()
; CFG-LABEL: staging.interior:
; CFG: br label %stage.store.interior
; CFG-LABEL: stage.store.interior:
; CFG: br label %barrier

; CFG-LABEL: define amdgpu_kernel void @canonical_k_loop(
; CFG-LABEL: dispatch:
; CFG: %interior.k.prefix.bound = and i32 %k, -32
; CFG: br i1 %interior.k.full, label %k.preheader.interior, label %k.preheader
; CFG-LABEL: k.preheader:
; CFG: %i.tail.start = phi i32 [ 0, %dispatch ], [ %next.interior, %k.latch.interior ]
; CFG: %acc.tail.start = phi i32 [ 0, %dispatch ], [ %acc.next.interior, %k.latch.interior ]
; CFG: %[[TAIL_REMAINDER:.*]] = icmp ne i32 %interior.k.prefix.bound, %k
; CFG: %[[NO_PREFIX:.*]] = xor i1 %interior.k.full, true
; CFG: %[[HAS_TAIL:.*]] = or i1 %[[NO_PREFIX]], %[[TAIL_REMAINDER]]
; CFG: br i1 %[[HAS_TAIL]], label %k.header, label %k.exit
; CFG-LABEL: k.body:
; CFG: call void @llvm.amdgcn.s.barrier()
; CFG-LABEL: k.body.interior:
; CFG: call void @llvm.amdgcn.s.barrier()
; CFG-LABEL: k.exit:
; CFG: %result.lcssa = phi i32 [ %next, %k.latch ], [ %i.tail.start, %k.preheader ]
; CFG: %acc.lcssa = phi i32 [ %acc.next, %k.latch ], [ %acc.tail.start, %k.preheader ]

; CFG-LABEL: define amdgpu_kernel void @canonical_synthesized_k_loop(
; CFG-LABEL: k.preheader:
; CFG: %interior.m.full = icmp sle i32 %y.base, %{{.*}}
; CFG: %interior.n.full = icmp sle i32 %x.base, %{{.*}}
; CFG: %interior.full = and i1 %interior.mn.full, %out.aligned
; CFG: br i1 %interior.k.full, label %k.preheader.split.interior, label %k.preheader.split
; CFG-LABEL: k.body.interior:
; CFG: br label %k.n.check.interior
; CFG-LABEL: k.n.check.interior:
; CFG: br label %k.k.check.interior
; CFG-LABEL: k.k.check.interior:
; CFG: br label %k.barrier.interior

; CFG-LABEL: define amdgpu_kernel void @canonical_predecessor_setup_k_loop(
; CFG-LABEL: k.preheader:
; CFG: %interior.m.full = icmp sle i32 %y.base, %{{.*}}
; CFG: %interior.n.full = icmp sle i32 %x.base, %{{.*}}
; CFG: %interior.full = and i1 %interior.mn.full, %out.aligned
; CFG: br i1 %interior.k.full, label %k.preheader.split.interior, label %k.preheader.split
; CFG-LABEL: k.body.interior:
; CFG: br label %k.n.check.interior
; CFG-LABEL: k.n.check.interior:
; CFG: br label %k.k.check.interior
; CFG-LABEL: k.k.check.interior:
; CFG: br label %k.barrier.interior

; CFG-LABEL: define amdgpu_kernel void @canonical_nested_staging_k_loop(
; CFG-LABEL: stage.m.check.interior:
; CFG: br label %stage.n.check.interior
; CFG-LABEL: stage.n.check.interior:
; CFG: br label %stage.k.check.interior
; CFG-LABEL: stage.k.check.interior:
; CFG: br label %stage.barrier.interior
; CFG-LABEL: stage.n.check:
; CFG: %or.cond = select i1 %k.in.bounds, i1 %n.in.bounds, i1 false
; CFG: br i1 %or.cond, label %stage.k.check, label %stage.latch
