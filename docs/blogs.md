---
title: All Blogs

nav_order: 9
description: Personal blogs, tutorials, and thoughts on AI, data science, and career.
---

<style>
.blog-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.blog-tag {
  background-color: #eef;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 0.8rem;
  white-space: nowrap;
}
</style>


# Blogs

Welcome to my blog corner! Here, I share hands-on tutorials, insights from real-world data science projects, and personal reflections on my journey in AI. I hope you find something helpful, or inspiring along the way.



<!-- <span class="label label-green">New (v0.4.0)</span> -->
<br>


<div style="display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 2rem;">
  <img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcb35b6dc-a7e7-42ef-b0a5-8ea42c4ffbc3_959x755.png" alt="MCp Thumbnail" style="width: 200px; border-radius: 8px;" />
  <div>
    <h3 style="margin-top: 0;"><a href="agentic-ai/2025-04-05-mcp.html">Understanding Model Context Protocol (MCP)</a></h3>
    <p>
    Think of the Model Context Protocol (MCP) as the USB-C of AI — a universal connector that lets language models securely plug into tools, data, and workflows with consistency and control.
    </p>
    <!-- Tags -->
    <div class="blog-tags">
    <span class="blog-tag">MCP Server</span>
    <span class="blog-tag">MCP Client</span>
    <span class="blog-tag">MCP Architecture</span>
    <span class="blog-tag">Tools</span>
    <span class="blog-tag">Agentic AI</span>
    </div>
  </div>
</div>

---

<div style="display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 2rem;">
  <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*OVi8blLZw_wf2rrxdlfbdg.png" alt="CLIP Thumbnail"
    style="width: 200px; height: 120px; object-fit: cover; border-radius: 8px;" />
  <div>
    <h3 style="margin-top: 0;"><a href="generative-ai/2025-02-21-clip-from-scratch.html">CLIP: A Vision + Language Model</a></h3>
    <p>A hands-on walkthrough of building a CLIP model, including vision and text encoders, contrastive loss, and training. Great for understanding how multimodal models really work.</p>
    <!-- Tags -->
    <div class="blog-tags">
    <span class="blog-tag">CLIP</span>
    <span class="blog-tag">Multimodal Model</span>
    <span class="blog-tag">Contrastive Learning</span>
    <span class="blog-tag">Multi-head Attention</span>
    </div>
  </div>
</div>

---

<div style="display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 2rem;">
  <img src="https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F493b62da-91e6-4f6b-896f-4af8c5eb1405_580x514.png" alt="CLIP Thumbnail" style="width: 200px; border-radius: 8px;" />
  <div>
    <h3 style="margin-top: 0;"><a href="generative-ai/2025-04-05-finetune-with-lora.html">How I Fine-Tuned Mistral-7B Model with LoRA (Low Rank Adaptation)</a></h3>
    <p>
    This blog walks through the post-training process of large language models (LLMs), focusing on Supervised Fine-Tuning (SFT) and why Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA are becoming essential. 
    <!-- We explore the core concepts, mathematical intuition (including the Eckart–Young theorem and Frobenius norm), and practical benefits of LoRA for transformer-based models. The post concludes with a step-by-step code walkthrough of fine-tuning the Mistral 7B model using LoRA, complete with training setup, inference, and tracking using Weights & Biases. -->
    </p>
    <!-- Tags -->
    <div class="blog-tags">
    <span class="blog-tag">Supervised Fine-Tuning (SFT)</span>
    <span class="blog-tag">Parameter-Efficient Fine-Tuning (PEFT)</span>
    <span class="blog-tag">Mistral-7B</span>
    <span class="blog-tag">LoRA</span>
    </div>
  </div>
</div>

---

<div style="display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 2rem;">
  <img src="/assets/images/yolo/9.png" alt="CLIP Thumbnail" style="width: 200px; border-radius: 8px;" />
  <div>
    <h3 style="margin-top: 0;"><a href="computer-vision/yolo-object-detection.html">YOLOv5 Custom Object Detection on Fashion Accessories</a></h3>
    <p>
        This project covers the complete pipeline of training a YOLOv5 model for custom object detection on fashion classes like topwear, bottomwear, footwear, handbag and eyewear. It walks through data annotation using Label Studio, training YOLOv5-Large with custom classes, and running inference using OpenCV-DNN. Whether you're new to object detection or fine-tuning YOLOv5 for a domain-specific task, this guide has you covered.
    </p>

    <div class="blog-tags">
    <span class="blog-tag">Object Detection</span>
    <span class="blog-tag">YOLOv5</span>
    <span class="blog-tag">Computer Vision</span>
    <span class="blog-tag">Data Annotation</span>
    <span class="blog-tag">Model Training</span>
    <span class="blog-tag">OpenCV</span>
    </div>
  </div>
</div>

---

<div style="display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 2rem;">
  <img src="https://www.gradio.app/_app/immutable/assets/header-image.DJQS6l6U.jpg" alt="gradio Thumbnail" style="width: 200px; border-radius: 8px;" />
  <div>
    <h3 style="margin-top: 0;"><a href="mlops/deployment-with-gradio.html">Build and Deploy PyTorch Models with Gradio & TorchScript</a></h3>
    <p>
        This blog covers how to create interactive demos and deploy deep learning models using Gradio and TorchScript. Starting from simple examples to advanced use cases like image classification, batch processing, and chatbot deployment, you'll learn how to build both server and client apps. We also explore how to serialize PyTorch models using TorchScript via scripting and tracing, and walk through deploying a real-world model end-to-end.
    </p>
    <div class="blog-tags">
        <span class="blog-tag">Model Deployment</span>
        <span class="blog-tag">Gradio</span>
        <span class="blog-tag">TorchScript</span>
        <span class="blog-tag">PyTorch</span>
        <span class="blog-tag">MLOps</span>
    </div>
  </div>
</div>

