# ClusterMark Quality Modifiers Implementation Plan

## Overview

This document provides a detailed plan to add quality modifier support to ClusterMark, enabling annotators to mark faces with quality attributes (`@poor`, `@blurry`, `@dark`, `@profile`, `@back`).

**Problem**: Current ClusterMark uses dropdown menus for character labels only. Annotators cannot specify that a face is poor quality (blurry, dark, bad angle) which should be down-weighted during cluster refinement.

**Solution**: Add quality attribute checkboxes to the ClusterMark UI, store them in the database, and export them in the JSON output.

---

## Current ClusterMark Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ClusterMark Stack                        │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React)        │  Backend (FastAPI/Python)        │
│  - Annotation UI        │  - REST API                       │
│  - Character dropdown   │  - PostgreSQL database            │
│  - Outlier handling     │  - JSON export                    │
├─────────────────────────────────────────────────────────────┤
│  Data Flow                                                     │
│  1. User selects character from dropdown                       │
│  2. Frontend sends POST /api/annotations                       │
│  3. Backend saves to PostgreSQL                               │
│  4. User exports → Backend generates JSON                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Proposed Changes

### Phase 1: Database Schema (PostgreSQL)

**Location**: `clustermark/backend/database.sql` or migration files

#### Current Schema (simplified)
```sql
CREATE TABLE annotations (
    id SERIAL PRIMARY KEY,
    episode_name VARCHAR(255),
    cluster_name VARCHAR(255),
    label VARCHAR(50),              -- Character name
    image_count INTEGER,
    outliers JSONB                  -- Array of {image, label}
);
```

#### New Schema
```sql
-- Add quality attributes table
CREATE TABLE quality_attributes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(20) UNIQUE NOT NULL,  -- '@poor', '@blurry', etc.
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Insert default quality attributes
INSERT INTO quality_attributes (name, description) VALUES
    ('@poor', 'Low quality face - blurry, dark, or extreme angle'),
    ('@blurry', 'Motion blur or out of focus'),
    ('@dark', 'Poorly lit face'),
    ('@profile', 'Side view or extreme angle'),
    ('@back', 'Back of head or not visible');

-- Add quality column to annotations table
ALTER TABLE annotations ADD COLUMN quality_attributes INTEGER[];

-- Add quality column to outliers (JSONB structure)
-- No change needed - outliers already use JSONB which can store any structure
```

#### Alternative: Store Quality in Outliers JSONB (Simpler)
```sql
-- No schema change needed! Just update JSON structure:

-- Current outlier format:
-- {"image": "scene_0_track_1_frame_001.jpg", "label": "Chandler"}

-- New format with quality:
-- {"image": "scene_0_track_1_frame_001.jpg", "label": "Chandler", "quality": ["@poor", "@blurry"]}

-- For main cluster images, add new column:
ALTER TABLE annotations ADD COLUMN main_quality JSONB DEFAULT '[]'::jsonb;
```

**Recommendation**: Use the simpler JSONB approach for main cluster images. No new table needed.

---

### Phase 2: Backend API Changes (FastAPI)

**Location**: `clustermark/backend/main.py` or relevant API files

#### 2.1 Update Data Models

```python
# clustermark/backend/models.py
from pydantic import BaseModel
from typing import List, Optional

class AnnotationRequest(BaseModel):
    episode_name: str
    cluster_name: str
    label: str
    image_count: int
    outliers: List[dict] = []  # Existing: [{"image": "...", "label": "..."}]
    quality_attributes: List[str] = []  # NEW: ["@poor", "@blurry"]

    class Config:
        schema_extra = {
            "example": {
                "episode_name": "Friends_S01E05",
                "cluster_name": "cluster-01",
                "label": "Rachel",
                "image_count": 20,
                "outliers": [],
                "quality_attributes": []  # Main cluster has no quality issues
            }
        }

class OutlierRequest(BaseModel):
    image: str
    label: str
    quality_attributes: List[str] = []  # NEW: Quality for individual outliers

    class Config:
        schema_extra = {
            "example": {
                "image": "scene_0_track_1_frame_001.jpg",
                "label": "Chandler",
                "quality_attributes": ["@blurry"]  # This outlier is blurry
            }
        }
```

#### 2.2 Update POST Endpoint

```python
# clustermark/backend/main.py

@app.post("/api/annotations")
async def save_annotation(annotation: AnnotationRequest):
    """
    Save annotation with optional quality attributes.

    New: quality_attributes field added to support @poor, @blurry, etc.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Insert with quality attributes
    cursor.execute("""
        INSERT INTO annotations (episode_name, cluster_name, label, image_count, outliers, main_quality)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (episode_name, cluster_name)
        DO UPDATE SET
            label = EXCLUDED.label,
            outliers = EXCLUDED.outliers,
            main_quality = EXCLUDED.main_quality
        RETURNING id
    """, (
        annotation.episode_name,
        annotation.cluster_name,
        annotation.label,
        annotation.image_count,
        json.dumps(annotation.outliers),
        json.dumps(annotation.quality_attributes)  # NEW
    ))

    conn.commit()
    cursor.close()
    conn.close()

    return {"status": "success", "id": cursor.fetchone()[0]}
```

#### 2.3 Update Export Endpoint

```python
# clustermark/backend/main.py

@app.get("/api/export/{episode_name}")
async def export_annotations(episode_name: str):
    """
    Export annotations with quality attributes included.

    New JSON format includes quality field for each cluster and outlier.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT cluster_name, label, image_count, outliers, main_quality
        FROM annotations
        WHERE episode_name = %s
    """, (episode_name,))

    annotations = {}
    for row in cursor.fetchall():
        cluster_name, label, image_count, outliers, main_quality = row

        # Parse outliers and add quality if present
        outliers_data = json.loads(outliers) if outliers else []

        annotations[cluster_name] = {
            "label": label,
            "image_count": image_count,
            "quality": json.loads(main_quality) if main_quality else [],  # NEW
            "outliers": outliers_data
        }

    cursor.close()
    conn.close()

    return {
        "episode_name": episode_name,
        "annotations": annotations,
        "version": "2.0"  # Indicate new format with quality support
    }
```

---

### Phase 3: Frontend UI Changes (React)

**Location**: `clustermark/frontend/src/components/`

#### 3.1 Quality Attribute Selector Component

**New file**: `clustermark/frontend/src/components/QualitySelector.jsx`

```jsx
import React, { useState } from 'react';

const QUALITY_OPTIONS = [
  { value: '@poor', label: 'Poor Quality', description: 'Blurry, dark, or extreme angle' },
  { value: '@blurry', label: 'Blurry', description: 'Motion blur or out of focus' },
  { value: '@dark', label: 'Dark', description: 'Poorly lit' },
  { value: '@profile', label: 'Profile', description: 'Side view' },
  { value: '@back', label: 'Back', description: 'Back of head' },
];

function QualitySelector({ selected, onChange, disabled = false }) {
  const handleToggle = (value) => {
    if (disabled) return;

    if (selected.includes(value)) {
      onChange(selected.filter(item => item !== value));
    } else {
      onChange([...selected, value]);
    }
  };

  return (
    <div className="quality-selector">
      <label className="text-sm font-medium text-gray-700">
        Quality Attributes (optional)
      </label>
      <div className="flex flex-wrap gap-2 mt-2">
        {QUALITY_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => handleToggle(option.value)}
            disabled={disabled}
            title={option.description}
            className={`
              px-3 py-1 rounded-full text-sm border transition-colors
              ${selected.includes(option.value)
                ? 'bg-orange-100 border-orange-500 text-orange-700'
                : 'bg-gray-50 border-gray-300 text-gray-600 hover:bg-gray-100'}
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            {option.label}
          </button>
        ))}
      </div>
      {selected.length > 0 && (
        <p className="text-xs text-gray-500 mt-1">
          Selected: {selected.join(', ')}
        </p>
      )}
    </div>
  );
}

export default QualitySelector;
```

#### 3.2 Update Annotation Page

**File**: `clustermark/frontend/src/pages/AnnotationPage.jsx`

```jsx
import React, { useState } from 'react';
import QualitySelector from '../components/QualitySelector';

function AnnotationPage() {
  const [selectedCharacter, setSelectedCharacter] = useState('');
  const [qualityAttributes, setQualityAttributes] = useState([]);
  const [outlierQuality, setOutlierQuality] = useState({});  // {imageIndex: ['@poor']}

  // ... existing code ...

  return (
    <div className="annotation-page">
      {/* Character selection dropdown - existing */}
      <select value={selectedCharacter} onChange={handleCharacterSelect}>
        <option value="">Select Character</option>
        <option value="Rachel">Rachel</option>
        <option value="Monica">Monica</option>
        {/* ... other characters ... */}
      </select>

      {/* NEW: Quality selector for main cluster */}
      <QualitySelector
        selected={qualityAttributes}
        onChange={setQualityAttributes}
        disabled={!selectedCharacter}
      />

      {/* Outlier labeling section - existing */}
      {outliers.map((outlier, index) => (
        <div key={index} className="outlier-item">
          <img src={outlier.image} alt={`Outlier ${index}`} />

          {/* Outlier character dropdown - existing */}
          <select
            value={outlier.label}
            onChange={(e) => handleOutlierLabelChange(index, e.target.value)}
          >
            <option value="">Select Character</option>
            <option value="Rachel">Rachel</option>
            {/* ... other characters ... */}
          </select>

          {/* NEW: Quality selector for individual outlier */}
          <QualitySelector
            selected={outlier.quality || []}
            onChange={(qualities) => handleOutlierQualityChange(index, qualities)}
          />
        </div>
      ))}

      <button onClick={handleSave}>
        Save Annotation
      </button>
    </div>
  );

  function handleSave() {
    // Prepare outliers with quality
    const outliersWithQuality = outliers.map(outlier => ({
      image: outlier.image,
      label: outlier.label,
      quality: outlier.quality || []  // NEW: Include quality
    }));

    // Send to API
    fetch('/api/annotations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        episode_name: episodeName,
        cluster_name: clusterName,
        label: selectedCharacter,
        image_count: images.length - outliers.length,
        outliers: outliersWithQuality,
        quality_attributes: qualityAttributes  // NEW: Main cluster quality
      })
    });
  }
}
```

#### 3.3 CSS Styling

**File**: `clustermark/frontend/src/components/QualitySelector.css` (or use Tailwind classes as shown above)

```css
.quality-selector {
  margin: 1rem 0;
  padding: 1rem;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  background-color: #f9fafb;
}

.quality-selector button {
  transition: all 0.2s ease;
}

.quality-selector button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
```

---

### Phase 4: Update Refinement Script

**File**: `src/cluster_refiner.py` (already partially done)

The refinement script needs to read the new JSON format:

```python
# Current JSON format:
{
  "cluster-01": {
    "label": "Rachel",
    "image_count": 20,
    "outliers": [
      {"image": "scene_0_track_1_frame_001.jpg", "label": "Chandler"}
    ]
  }
}

# New JSON format:
{
  "cluster-01": {
    "label": "Rachel",
    "image_count": 20,
    "quality": [],  // NEW: Quality for main cluster images
    "outliers": [
      {
        "image": "scene_0_track_1_frame_001.jpg",
        "label": "Chandler",
        "quality": ["@blurry"]  // NEW: Quality for this outlier
      }
    ]
  }
}
```

**Update needed in `cluster_refiner.py`**:

```python
def _extract_ground_truth(self) -> Dict[str, Any]:
    # ... existing code ...

    for cluster_key, cluster_info in self.annotations['cluster_annotations'].items():
        # Parse label and quality from main cluster
        raw_label = cluster_info['label'].lower()
        main_label, main_modifiers = self._parse_label_with_modifiers(raw_label)

        # NEW: Also check for quality field
        quality_field = cluster_info.get('quality', [])
        if quality_field:
            # Quality from annotation JSON (new format)
            main_quality = set(quality_field)
        else:
            # Legacy: parse from label string
            main_quality = main_modifiers

        # Process main cluster images with quality
        for img_path in cluster_info.get('image_paths', []):
            filename = os.path.basename(img_path)
            norm_key = self._normalize_filename_key(filename)

            if norm_key not in self.image_mapping_norm:
                continue

            face_data = self.image_mapping_norm[norm_key]
            face_id = face_data['unique_face_id']

            # NEW: Set weight based on quality
            if main_quality & constants.QUALITY_MODIFIERS:
                face_weights[face_id] = 0.5

        # Process outliers with quality
        for outlier in cluster_info.get('outliers', []):
            raw_label = outlier['label'].lower()
            outlier_label, outlier_modifiers = self._parse_label_with_modifiers(raw_label)

            # NEW: Check for quality field in outlier
            outlier_quality_field = outlier.get('quality', [])
            if outlier_quality_field:
                outlier_quality = set(outlier_quality_field)
            else:
                outlier_quality = outlier_modifiers

            # Store weight
            if outlier_quality & constants.QUALITY_MODIFIERS:
                face_weights[face_id] = 0.5
```

---

## Implementation Steps

### Step 1: Backend Changes (1-2 hours)
1. Update database schema (add `main_quality` column)
2. Update Pydantic models to include `quality_attributes`
3. Update POST `/api/annotations` endpoint
4. Update GET `/api/export/{episode_name}` endpoint
5. Test with curl/Postman

### Step 2: Frontend Changes (2-3 hours)
1. Create `QualitySelector.jsx` component
2. Add to annotation page for main cluster
3. Add to outlier labeling section
4. Style with CSS/Tailwind
5. Test UI interactions

### Step 3: Update Refinement Script (30 minutes)
1. Update `_extract_ground_truth()` to read new JSON format
2. Test with sample annotations

### Step 4: End-to-End Testing (1 hour)
1. Full annotation workflow with quality attributes
2. Export and verify JSON format
3. Run refinement script with new annotations
4. Verify poor quality faces are down-weighted

---

## Migration Strategy

### For Existing Annotations

Existing annotations won't have quality fields. The refinement script should handle both formats:

```python
# Backward compatible read
quality_field = cluster_info.get('quality')
if quality_field is None:
    # Old format - no quality field
    quality_set = set()
else:
    # New format - quality field present
    quality_set = set(quality_field)
```

### Database Migration

```sql
-- Add new column with default empty array
ALTER TABLE annotations ADD COLUMN main_quality JSONB DEFAULT '[]'::jsonb;

-- Existing annotations get empty quality (no change in behavior)
-- New annotations can have quality attributes
```

---

## Testing Checklist

### Backend
- [ ] POST `/api/annotations` accepts `quality_attributes` field
- [ ] GET `/api/export/{episode_name}` includes quality in output
- [ ] Database stores quality correctly
- [ ] Empty quality array handled properly

### Frontend
- [ ] Quality buttons toggle on/off
- [ ] Selected qualities displayed correctly
- [ ] Quality saved with annotation
- [ ] Quality loaded when editing existing annotation
- [ ] Disabled state works when no character selected

### Integration
- [ ] Full annotation workflow with quality
- [ ] Export JSON has correct format
- [ ] Refinement script reads new format
- [ ] Backward compatible with old annotations

---

## File Changes Summary

| File | Change | Lines Added |
|------|--------|-------------|
| `backend/database.sql` | Add `main_quality` column | 5 |
| `backend/models.py` | Add quality fields to Pydantic models | ~20 |
| `backend/main.py` | Update API endpoints | ~40 |
| `frontend/src/components/QualitySelector.jsx` | New component | ~80 |
| `frontend/src/pages/AnnotationPage.jsx` | Integrate quality selector | ~50 |
| `src/cluster_refiner.py` | Read new JSON format | ~30 |

**Total**: ~225 lines of code

---

## Alternative: Simpler Approach (No ClusterMark Changes)

If modifying ClusterMark is too complex, use this workaround:

1. **Add quality labels to ClusterMark dropdown**
   - Add special labels like: `dk_poor`, `rachel_poor`, `monica_poor`
   - Use these when face is poor quality

2. **Parse quality from label in refinement script**
   ```python
   def _parse_label_with_quality(self, label: str) -> Tuple[str, Set[str]]:
       """Parse 'rachel_poor' -> ('rachel', {'@poor'})"""
       if '_poor' in label:
           base_label = label.split('_')[0]
           return base_label, {'@poor'}
       return label, set()
   ```

3. **Pros**: No ClusterMark changes needed
4. **Cons**: Less elegant, requires special labels for each character

---

## Timeline Estimate

| Phase | Time | Dependencies |
|-------|------|--------------|
| Database schema | 30 min | None |
| Backend API | 1-2 hours | Database schema |
| Frontend UI | 2-3 hours | Backend API |
| Refinement update | 30 min | Frontend UI |
| Testing | 1 hour | All above |
| **Total** | **5-7 hours** | None |

---

## Next Steps

1. **Confirm approach**: Get approval on JSONB storage vs separate table
2. **Set up development environment**: Clone ClusterMark locally
3. **Create feature branch**: `git checkout -b feature/quality-attributes`
4. **Implement in phases**: Backend → Frontend → Integration → Testing
5. **Deploy to staging**: Test with sample episode
6. **Production deployment**: Update annotation workflow docs

---

## References

- ClusterMark repo: https://github.com/yibeichan/clustermark
- Current refinement code: `src/cluster_refiner.py`
- Constants: `src/constants.py` (QUALITY_MODIFIERS)
- Related docs: `docs/refinement_redesign_plan.md`
