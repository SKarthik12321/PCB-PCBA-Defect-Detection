# PCB/PCBA Defect Detection Architecture

This document contains the architecture diagram for the Lightweight PCB/PCBA Defect Detection model, utilizing Adaptive Dual-Attentive Feature Recalibration.

The diagram is written in [Mermaid](https://mermaid.js.org/) and can be previewed in any Markdown viewer that supports Mermaid rendering (such as GitHub, VS Code, or Typora).

## Architecture Flow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'primaryTextColor': '#000000', 'lineColor': '#000000', 'fontFamily': 'arial'}}}%%
graph TD
    %% Base Inputs
    In_Image["Input PCB/PCBA Image"]
    Preproc["Dataset Preprocessing Engine<br>(Augmentation & Normalization)"]
    
    In_Image -.-> Preproc
    
    %% Backbone and Recalibration Output
    Backbone["Modified Lightweight Backbone"]
    DualAttn["Dual-Attentive Channel-Spatial Recalibration"]
    
    Preproc ==> Backbone
    Preproc ==> DualAttn

    %% Architecture Flow
    subgraph Backbone_Features ["Backbone Features"]
        direction LR
        subgraph Shallow ["Shallow Features"]
            direction TB
            C1["Conv I"] --> P1["Pool I"]
            C2["Conv II"] --> P2["Pool II"]
            C3["Conv III"] --> P3["Pool III"]
            C4["Conv IV"] --> P4["Pool IV"]
        end
        subgraph Deep ["Deep Features"]
            direction TB
            Conv4_5["Conv IV . . . Conv V"]
        end
        C4 -.-> Conv4_5
    end
    
    Backbone ==> Backbone_Features
    Backbone -- "Skip Connection" --> FusedMap
    
    %% Attention Interface
    subgraph AttnBox ["Channel Attention (SE) & Spatial Attention (SA)"]
        direction TB
        SA_SE["Attention Modulation"]
    end
    
    DualAttn --> AttnBox
    
    P1 --> SA_SE
    P2 --> SA_SE
    P3 --> SA_SE
    P4 --> SA_SE

    %% CARFT Neck Area
    subgraph Neck ["CARFT Neck"]
        direction TB
        AttnBlocks["Adaptive Transformer Blocks"]
        ResFusion["Residual Shallow Feature Fusion"]
        ResFusion -->|Up| AttnBlocks
    end
    
    SA_SE --> AttnBlocks
    
    %% Fusion
    FusedMap["Fused Multi-Scale Feature Map"]
    Conv4_5 ==> FusedMap
    FusedMap ---> ResFusion
    
    %% Final Heads
    LocHead["Adaptive Localization Head<br>(IoU Optimization)"]
    LossNet["Adaptive Gradient-Balanced N-CIoU Loss"]
    
    %% Flow to Heads
    AttnBlocks ==> LocHead
    ResFusion ==> LossNet
    
    %% Dashed Connections
    DualAttn -.->|"Dashed link"| LocHead
    C4 -.->|"Dashed link"| LossNet
    LossNet -.->|"Loss Feedback"| LocHead
    
    %% Output Section
    OutResult["PCB/PCBA Defect Detection Output<br>Missing Hole | Broken Wire | Sweeping Wire<br>56.7 | 93.5 | 72.5"]
    LocHead ==> OutResult
    
    BottomTitle["Architecture for Lightweight PCB/PCBA Defect Detection<br>with Adaptive Dual-Attentive Feature Recalibration"]
    BottomTitle -->|Upwards| OutResult
    
    %% Class Definitions
    classDef yellowNode fill:#fcf1b8,stroke:#b8990b,stroke-width:2px;
    classDef blueNode fill:#d6f4f5,stroke:#2d9ead,stroke-width:2px;
    classDef purpleNode fill:#7f5a9e,stroke:#3b1e6d,stroke-width:2px,color:#fff;
    classDef greenNode fill:#e9fce6,stroke:#5ba65c,stroke-width:2px;
    classDef redNode fill:#e59c99,stroke:#b03434,stroke-width:2px;
    classDef lossNode fill:#f5c9b6,stroke:#c4582f,stroke-width:2px;
    classDef titleNode fill:#ffebcd,stroke:#e6b800,stroke-width:2px;
    classDef dashedBox fill:none,stroke:#2d9ead,stroke-width:2px,stroke-dasharray: 5 5;
    
    class In_Image,Preproc,Backbone,LocHead yellowNode
    class C1,C2,C3,C4,P1,P2,P3,P4,Conv4_5,DualAttn,SA_SE blueNode
    class Neck,AttnBlocks,ResFusion purpleNode
    class FusedMap redNode
    class LossNet lossNode
    class OutResult greenNode
    class BottomTitle titleNode
    class AttnBox dashedBox
```
