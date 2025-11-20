# GPU Acceleration & Performance Optimization Plan

Branch: `gpu-accelerate-tracking-clustering`

## Executive Summary

Optimize steps 03 (tracking) and 04 (clustering) in the face processing pipeline to reduce processing time from ~1.5 hours to target ~20-30 minutes per episode.

## Current Performance Issues

### Step 03: Within-Scene Tracking
**Bottleneck:** Frame selection with random video seeking (~40-50 min per episode)
- Current: Random seeks to 300+ frame positions
- Problem: Video codecs must decode from keyframes (expensive)
- GPU: Not utilized (CPU-only operations)

### Step 04: Face Clustering
**Bottleneck:** Sequential embedding extraction (~10-15 min per episode)
- Current: Process one image at a time through FaceNet
- Problem: Poor GPU utilization (batch size = 1)
- GPU: Used but underutilized

---

## Optimization 1: Sequential Frame Reading (Step 03)

### Current Implementation
```python
# src/face_tracker.py:224
for entry in face_group:
    frame_idx = entry['frame']
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # SLOW!
    ret, frame = cap.read()
    # Process face...
```

### Optimized Implementation
```python
# Strategy: Read video once, process all faces sequentially
# 1. Collect all needed frames with face metadata
# 2. Sort by frame index
# 3. Read video once in sequential order
# 4. Process all faces for each frame as encountered
```

### Implementation Details

**Phase 1: Collect frame requirements**
- Build map: `frame_idx -> [(scene_id, face_id, face_coords), ...]`
- Stores which faces need which frames

**Phase 2: Sequential read**
- Sort frame indices
- Read video sequentially (no seeking)
- For each frame, process all faces that need it
- Cache frame in memory while processing multiple faces

**Phase 3: Score and select**
- Calculate quality scores during read
- Select top N frames per face after all frames processed

### Expected Impact
- **Speedup:** 5-10x for frame selection
- **Time saved:** ~35-40 minutes per episode
- **Tradeoff:** Slightly higher memory usage (frame cache)

### Files to Modify
- `src/face_tracker.py` - FrameSelector class
  - Refactor `select_top_frames_per_face()` method
  - Add helper methods: `_collect_frame_requirements()`, `_sequential_read_and_score()`, `_select_best_frames()`

---

## Optimization 2: Batch Embedding Extraction (Step 04)

### Current Implementation
```python
# src/face_clusterer.py:32-50
for scene_id, faces in selected_frames.items():
    for face_data in faces:
        for frame_info in face_data['top_frames']:
            face_tensor = self.load_image(image_path)  # One at a time
            embedding = self.model(face_tensor).cpu().numpy()
```

### Optimized Implementation
```python
# Strategy: Batch process images for better GPU utilization
# 1. Collect all image paths
# 2. Load images in batches (32-64)
# 3. Process batch through model once
# 4. Map embeddings back to faces
```

### Implementation Details

**Batch Configuration**
- Batch size: 32 (balance between memory and throughput)
- Use PyTorch DataLoader with num_workers=4 for parallel I/O
- Prefetch next batch while GPU processes current batch

**Batching Strategy**
- Flatten all images into single list with metadata
- Group into batches of fixed size
- Process each batch through model
- Reconstruct hierarchical structure after processing

**Memory Management**
- Pin memory for faster CPU->GPU transfer
- Clear cache between large batches
- Monitor GPU memory usage

### Expected Impact
- **Speedup:** 2-4x for embedding extraction
- **Time saved:** ~7-12 minutes per episode
- **GPU utilization:** 20% -> 80%+

### Files to Modify
- `src/face_clusterer.py` - FaceEmbedder class
  - Add `get_face_embeddings_batch()` method
  - Add helper: `_create_batch_dataloader()`
  - Keep old method for backward compatibility

---

## Optimization 3: Additional Minor Optimizations

### A. Vectorized IoU Calculation (Step 03)
**Optional - implement if profiling shows bottleneck**

Current: Loop through tracks one-by-one
```python
for track in tracks:
    iou = self.calculate_iou(...)  # Per track
```

Optimized: Vectorize with NumPy broadcasting
```python
# Calculate IoU for all active tracks at once
ious = vectorized_iou(detection_box, all_track_boxes)  # Vectorized
best_idx = np.argmax(ious)
```

**Expected impact:** 1.5-2x speedup for IoU matching (minor overall impact)

### B. Image Decoding Optimization (Both Steps)
- Use cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION for faster loading
- Consider using Pillow-SIMD for faster JPEG decoding
- Pre-allocate image buffers where possible

---

## Implementation Plan

### Phase 1: Step 03 Sequential Reading ⭐ **Highest Priority**
1. Refactor FrameSelector.select_top_frames_per_face()
2. Add sequential reading logic
3. Test on single episode (friends_s01e01a)
4. Benchmark: Compare before/after timing

**Estimated dev time:** 2-3 hours
**Risk:** Medium (significant refactor, but clear logic)

### Phase 2: Step 04 Batch Embeddings ⭐ **High Priority**
1. Add FaceEmbedder.get_face_embeddings_batch()
2. Create DataLoader wrapper for image loading
3. Update 04_face_clustering.py to use batch method
4. Test on single episode
5. Benchmark GPU utilization

**Estimated dev time:** 1-2 hours
**Risk:** Low (additive change, can keep old method)

### Phase 3: Testing & Validation
1. Run full pipeline on test episode
2. Verify outputs match original (bit-for-bit or very close)
3. Profile to identify any remaining bottlenecks
4. Document performance improvements

**Estimated dev time:** 1 hour
**Risk:** Low

### Phase 4: Optional Optimizations (if needed)
1. Vectorized IoU (if still bottleneck)
2. Image decoding optimizations
3. Memory profiling and optimization

---

## Success Metrics

### Performance Targets
- **Step 03:** 40-50 min → 5-8 min (5-10x improvement)
- **Step 04:** 10-15 min → 3-5 min (3-4x improvement)
- **Total pipeline (01-04b):** ~1.5 hours → ~20-30 min

### Validation Criteria
- Output JSON files must match original (or negligible differences due to floating point)
- No regression in quality metrics
- GPU memory usage stays within limits (≤16GB)
- Code remains readable and maintainable

---

## Risk Mitigation

### Backward Compatibility
- Keep old methods with deprecation warnings
- Add feature flags to toggle optimizations
- Extensive testing before replacing old code

### Memory Management
- Monitor peak memory usage during development
- Add configurable batch sizes
- Implement graceful degradation if OOM

### Testing Strategy
- Unit tests for new functions
- Integration test on single episode
- Comparison test: old vs new outputs
- Performance benchmarks

---

## Files to Modify

### Core Changes
- `src/face_tracker.py` - FrameSelector refactor
- `src/face_clusterer.py` - Add batch processing

### Scripts (minimal changes)
- `scripts/03_within_scene_tracking.py` - Use new FrameSelector API (if changed)
- `scripts/04_face_clustering.py` - Switch to batch embedding method

### Testing
- `tests/test_face_tracker.py` - Add tests for sequential reading
- `tests/test_face_clusterer.py` - Add tests for batch processing

### Documentation
- Update CLAUDE.md with new performance characteristics
- Add performance benchmarking script

---

## Next Steps

1. ✅ Create this optimization plan
2. ⏳ Implement sequential frame reading (Step 03)
3. ⏳ Implement batch embeddings (Step 04)
4. ⏳ Test and benchmark on friends_s01e01a
5. ⏳ Create PR with performance comparisons
6. ⏳ Update documentation

---

## Notes

- Both optimizations are independent and can be implemented in parallel
- Sequential reading has higher impact but is more complex
- Batch embeddings is simpler and still provides good gains
- Consider implementing batch embeddings first if wanting quick wins
