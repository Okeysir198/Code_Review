# Status Update Report
**Date:** June 17, 2025

## Call Center Project

**Current Focus:** Background noise cancellation, semantic turn detection, and user interruption capabilities

**Progress:**
- ✅ **Completed:** Noise cancellation and semantic turn detection modules
- 🔄 **In Progress:** Refining interruption behavior to enable seamless user interruptions
  - *Current limitation: System stops AI response when user speaks but then generates new response based on interruption content, creating unwanted conversation shifts*

**Demo:** [Testing Interface](https://f0359d170a103ec268.gradio.live/)

**This Week's Priorities:**
- Integrate completed modules into the audio streaming pipeline
- Resume development of AI agent behavior optimization for debt collection scenarios

---

## Multi-Camera Tracking System

**Demo:** [Full Pipeline Results - Retail Environment](https://f97f049ae38dc54ae7.gradio.live/)
*Demonstrates both cross-camera and single-camera tracking capabilities*

**Performance Metrics (Cross-Camera Tracking):**
- **ID Switch Rate:** 0.74 (Target: 1.0)
  - *Measures how well the system maintains the same ID for each person as they move between cameras*
  - *Formula: Total ground truth people ÷ Number of ID changes per person*
  - *Higher score = fewer unwanted ID changes*
- **Person Switch Rate:** 0.90 (Target: 1.0)  
  - *Measures how accurately each assigned ID represents only one person (no mixing)*
  - *Formula: Total ground truth IDs ÷ Number of person changes per ID*
  - *Higher score = each ID consistently tracks the same individual*

**This Week's Objectives:**
- **Robustness Testing:** continue testing pipeline in additional videos to validate performance consistency
- **Visualization Enhancement:** Optimize 2D layout display including camera positioning and shelf mapping
- **Error Analysis:** Systematically document and categorize all ID/person switch incidents for targeted improvements