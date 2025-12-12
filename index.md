---
title: About Me
# layout: minimal
layout: home
nav_order: 1
description: "Just the Docs is a responsive Jekyll theme with built-in search that is easily customizable and hosted on GitHub Pages."
permalink: /
---

# Ankita Mungalpara

<br>

<img src="assets/images/profilepic.jpeg" alt="Example" style="float: left; margin-right: 1rem; width: 200px; border-radius: 8px;" />

<p style="text-align: justify;">
I’m a <span class="text-purple-100">researcher and data scientist</span> with hands-on experience in computer vision, deep learning, and generative AI. My current work focuses on advancing <span class="text-purple-100">agentic AI and multimodal large language models (LLMs)</span>. I hold a Master of Science in Data Science from the University of Massachusetts Dartmouth.
</p>

<p style="text-align: justify;">
With over <span class="text-purple-100">three years of industry experience</span> in data science and machine learning, I’ve built scalable, real-world AI solutions—from working as an <span class="text-purple-100">ML Engineer at Tiger Analytics</span> to developing a conversational LLM agent during my <span class="text-purple-100">Summer 2024 internship at Johnson & Johnson Innovative Medicine.</span>

I'm passionate about solving complex problems, pushing the boundaries of AI research, and transforming emerging ideas into impactful, real-world solutions.
</p>

<div style="text-align: center;">
<span style="display: inline-flex; align-items: center; gap: 20px; margin-top: 10px;">

  <a href="https://github.com/AnkitaMungalpara" target="_blank">
    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub" width="30" height="30" />
  </a>

  <a href="https://medium.com/@mungalpara.ankita" target="_blank">
    <img src="https://cdn-icons-png.flaticon.com/512/5968/5968906.png" alt="Medium" width="30" height="30" />
  </a>

  <a href="https://substack.com/@ankitamungalpara" target="_blank">
    <img src="https://substackcdn.com/image/fetch/$s_!SuE5!,w_60,h_60,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack.com%2Fimg%2Fsubstack-app-icon.png" alt="Substack" width="30" height="30" />

  <a href="https://www.linkedin.com/in/ankita-mungalpara" target="_blank" title="LinkedIn">
    <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn" width="30" height="30" />
  </a>
  </a>

</span>
</div>



<style>
  .site-logo {
  background-size: cover !important;
  background-position: center;
  width: 100px;
  height: 100px;
  border-radius: 50%;
}

  .floating-profile {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
    background: white;
    border-radius: 50%;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    padding: 4px;
  }
  
  .floating-profile img {
    width: 250px;
    height: 250px;
    border-radius: 50%;
    object-fit: cover;
  }

  .experience-section {
  display: flex;
  flex-direction: column;
  gap: 1.8rem; /* clean spacing between entries */
  padding: 1rem 0;
}

.experience-entry {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
  transition: all 0.3s ease-in-out;
  padding-bottom: 0.5rem;
  /* border-bottom: 1px solid rgba(0, 0, 0, 0.05); optional light separator line */
}


.experience-logo {
  width: 150px;        /* Set a fixed width */
  height: 100px;        /* Set a fixed height for rectangle */
  object-fit: contain; /* Ensure the logo scales nicely without distortion */
  border-radius: 8px;  /* Slight rounding for a polished look */
  background-color: white; /* Optional: makes logos with transparency look cleaner */
  box-shadow: 0 0 4px rgba(0,0,0,0.1); /* Optional subtle shadow */
}

.experience-text {
  flex: 1;
  min-width: 250px;
}

.experience-role {
  font-weight: bold;
  font-size: 1.1rem;
}

.experience-company {
  font-size: 1rem;
  color: #555;
}

.experience-date {
  font-size: 0.9rem;
  color: #777;
  margin-top: 2px;
}

.hover-details {
  opacity: 0;
  max-height: 0;
  overflow: hidden;
  transition: opacity 0.3s ease, max-height 0.5s ease;
  font-size: 0.95rem;
  color: #444;
  margin-top: 0.3rem;
}

.experience-entry:hover .hover-details {
  opacity: 1;
  max-height: 200px;
}

/* Mobile responsiveness */
@media (max-width: 600px) {
  .experience-entry {
    flex-direction: column;
    align-items: flex-start;
  }
  .experience-logo {
    margin-bottom: 0.5rem;
  }
}

.certification-card {
  display: flex;
  align-items: center;
  margin: 20px 0;
  transition: transform 0.3s ease;
}

.certification-card:hover {
  transform: scale(1.02);
}

.cert-link {
  text-decoration: none;
  color: inherit;
  display: flex;
  align-items: center;
  gap: 15px;
  flex-wrap: wrap;
}

.cert-logo {
  width: 100px;
  height: 100px;
  object-fit: contain;
  border-radius: 8px;
  box-shadow: 0 0 6px rgba(0, 0, 0, 0.1);
}

.cert-info {
  max-width: 500px;
}

.publication-card {
  display: flex;
  align-items: center;
  margin: 20px 0;
  transition: transform 0.3s ease;
}

.publication-card:hover {
  transform: scale(1.02);
}

.publication-link {
  text-decoration: none;
  color: inherit;
  display: flex;
  align-items: center;
  gap: 15px;
  flex-wrap: wrap;
}

.publication-logo {
  width: 90px;
  height: auto;
  object-fit: contain;
  border-radius: 6px;
  box-shadow: 0 0 6px rgba(0, 0, 0, 0.1);
}

.publication-info {
  max-width: 600px;
}


.projects-container {
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 20px;
}

.project-card {
  display: flex;
  align-items: flex-start;
  background: #fff;
  border-radius: 16px;
  padding: 20px;
  gap: 20px;
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.05);
  transition: all 0.3s ease;
  flex-wrap: wrap;
}

.project-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 16px 36px rgba(0, 0, 0, 0.1);
}

.project-logo-wrapper {
  min-width: 90px;
  height: 90px;
  border-radius: 12px;
  background: #f3f4f6;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: inset 2px 2px 6px #d1d5db, inset -2px -2px 6px #ffffff;
}

.project-logo {
  max-width: 150px;
  max-height: 150px;
  object-fit: contain;
}

.project-details {
  flex: 1;
}

.project-title {
  font-size: 1.25rem;
  margin: 0;
  background: linear-gradient(to right, #6366f1, #ec4899);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.project-subtitle {
  font-size: 0.95rem;
  color: #6b7280;
}

.project-description {
  margin: 8px 0;
  font-size: 0.95rem;
  color: #374151;
}

.tech-stack {
  font-size: 0.85rem;
  color: #9ca3af;
  font-family: monospace;
}

@media (max-width: 768px) {
  .project-card {
    flex-direction: column;
    align-items: flex-start;
  }

  .project-logo-wrapper {
    margin-bottom: 10px;
  }
}


.project-categories {
  display: flex;
  justify-content: center;
  gap: 24px;
  padding: 40px 20px;
  flex-wrap: wrap;
}


.category-tile {
  background: #f9fafb;
  border-radius: 1.25rem;
  padding: 2rem;
  text-align: center;
  font-weight: 600;
  font-size: 1.25rem;
  color: #111827;
  text-decoration: none;
  transition: all 0.3s ease;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
  width: 100%;
  max-width: 320px;
  height: 160px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}

.category-tile:hover {
  transform: translateY(-8px);
  box-shadow: 0 20px 30px rgba(0, 0, 0, 0.1);
  background: linear-gradient(to right, #6366f1, #ec4899);
  color: white;
}

.project-categories-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  padding: 2rem;
  justify-items: center;
}

  
/* CARD + IMAGE ------------------------------------------------------------ */
.blog-card {
  border-radius: 12px;
  overflow: hidden;
  background: #fff;
  box-shadow: 0 4px 10px rgba(0,0,0,.05);
  transition: transform .3s ease;
  margin-bottom: 2rem;
}
.blog-card:hover { transform: translateY(-6px); }

.blog-link  { display: flex; flex-direction: column; color: inherit; text-decoration: none; }
.blog-image { width: 100%; height: 180px; object-fit: cover; }

/* CONTENT ----------------------------------------------------------------- */
.blog-content { padding: 1rem 1.25rem 1.5rem; }

.blog-meta    { font-size: .85rem; color: #6b7280; margin: .25rem 0; }

/* TAG CHIPS --------------------------------------------------------------- */
.blog-tags      { margin: .5rem 0 .75rem; display: flex; flex-wrap: wrap; gap: .5rem; }
.tag            { background:#eef2ff; color:#4f46e5; font-size:.75rem; font-weight:600;
                  padding:.25rem .55rem; border-radius:9999px; }

/* CLAMPED PREVIEW --------------------------------------------------------- */
.blog-preview   { font-size:.95rem; color:#374151;
                  display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical;
                  overflow:hidden; }

/* READ-MORE ----------------------------------------------------------------*/
.read-more      { display:inline-block; margin-top:.9rem; color:#4f46e5; font-weight:500;
                  font-size:.9rem; transition:color .3s; }
.read-more:hover{ color:#4338ca; text-decoration:underline; }

/* GRID WRAPPER (if you show many cards) ----------------------------------- */
.blog-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:2rem; }


/* work ex */

.work-experience-section {
  padding: 3rem 1rem;
}

.experience-grid {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.experience-card {
  padding: 1.5rem;
  background: #f9f9f9;
  border-radius: 12px;
  transition: transform 0.3s ease;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.experience-card:hover {
  transform: translateY(-5px);
}

.experience-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
}

.company-logo {
  width: 60px;
  height: 60px;
  object-fit: contain;
  border-radius: 10px;
  background: white;
}

.position {
  font-style: italic;
  margin: 0.25rem 0;
}

.duration {
  font-size: 0.9rem;
  color: #666;
}

.experience-description p {
  margin: 0;
  overflow: hidden;
  max-height: 10.5em;
  line-height: 1.5em;
  text-overflow: ellipsis;
}


.description-content.collapsed {
  max-height: 3rem;
  overflow: hidden;
  transition: max-height 0.3s ease;
}

.description-content.expanded {
  max-height: none;
}

.read-more-toggle {
  display: inline-block;
  margin-top: 0.9rem;
  color: #4f46e5;
  font-weight: 500;
  font-size: 0.9rem;
  transition: color 0.3s;
  cursor: pointer;
}

.read-more-toggle:hover {
  color: #4338ca;
  text-decoration: underline;
}


/* education */
.education-section {
  padding: 3rem 1rem;
}

.section-title {
  font-size: 1.75rem;
  margin-bottom: 2rem;
  color: var(--body-text-color);
}

.education-grid {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.education-card {
  padding: 1.5rem;
  background: #f9f9f9;
  border-radius: 12px;
  transition: transform 0.3s ease;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.education-card:hover {
  transform: translateY(-5px);
}

.education-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
}

.university-logo {
  width: 60px;
  height: 60px;
  object-fit: contain;
  border-radius: 10px;
  background: white;
}

.university-name {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: #222;
}

.degree {
  font-style: italic;
  margin: 0.25rem 0;
  font-size: 0.95rem;
  color: #555;
}

.education-details p {
  margin: 0;
  line-height: 1.5;
  font-size: 0.92rem;
  color: #444;
}


 /* technical skills */

 .technical-skills {
  margin: 2rem 0;
  font-family: 'Inter', 'Helvetica Neue', sans-serif;
}

.technical-skills h2 {
  font-size: 1.5rem;
  margin-bottom: 1rem;
  color: #1f2937; /* Gray-800 */
}

.skill-group {
  margin-bottom: 1rem;
}

.skill-group strong {
  display: block;
  margin-bottom: 0.3rem;
  color: #374151; /* Gray-700 */
}

.badge {
  display: inline-block;
  background-color: #e0e7ff; /* Indigo-100 */
  color: #3730a3; /* Indigo-800 */
  padding: 0.25rem 0.5rem;
  margin: 0.25rem 0.4rem 0.25rem 0;
  border-radius: 0.5rem;
  font-size: 0.85rem;
  font-weight: 500;
  transition: background-color 0.3s ease;
}

.badge:hover {
  background-color: #c7d2fe; /* Indigo-200 */
}

/* section divider */
.section-divider {
  margin: 3rem 0;
  border-top: 1px solid #e5e7eb; /* light gray, modern neutral */
  opacity: 0.7;
}


</style>

<script>
  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.read-more-toggle').forEach(button => {
      button.addEventListener('click', () => {
        const content = button.previousElementSibling;
        content.classList.toggle('collapsed');
        content.classList.toggle('expanded');
        button.textContent = content.classList.contains('collapsed') ? 'Read more →' : 'Read less ↑';
      });
    });
  });
</script>


<script>
function scrollToSection(id) {
  const section = document.getElementById(id);
  if (section) {
    section.scrollIntoView({ behavior: 'smooth' });
  }
}
</script>



# All Projects

<br>

This is a collection of my hands-on projects in agentic AI, generative AI, LLMOps, computer vision, and MLOps. Each project jumps into a key concept or framework in AI and ML, often supported by practical implementations or tutorials.


1. [Multimodal Video RAG Agent](docs/agentic-ai/multimodel-video-agent.html)
2. [Model Context Protocol (MCP)](docs/agentic-ai/2025-04-05-mcp.html)
3. [Managing Memory, Context, and State in an AI Agent](docs/agentic-ai/memory_context_in_agent.html)
4. [Document Intelligence: Modern Approaches to Extracting Structured Information](docs/generative-ai/document-ai/index.html)
    - [Part 1: Document Parsing with PaperMage](docs/generative-ai/document-ai/01_document_parsing_with_papermage.html)
    - [Part 2: Document Parsing with DONUT](docs/generative-ai/document-ai/02_document_arsing_with_DONUT.html)
    - [Part 3: Document Parsing with Nougat](docs/generative-ai/document-ai/03_document_parsing_with_Nougat.html)
    - [Part 4: Document Parsing with GOT-OCR2.0](docs/generative-ai/document-ai/04_Document_understanding_with_GOT_OCR2.0.html)
    - [Part 5: Document Parsing with MierU](docs/generative-ai/document-ai/05_document_understanding_with_MinerU.html)
5. [Advanced RAG](docs/generative-ai/advanced-rag/index.html)
6. [Building CLIP From Scratch](docs/generative-ai/2025-02-21-clip-from-scratch.html)
7. [Fine-Tune Mistral-7B Model with LoRA: Sentiment Classification](docs/generative-ai/2025-04-05-finetune-with-lora.html)
8. [Fine-Tune Mistral-7B Model with QLoRA: Financial Q&A](docs/generative-ai/finetune-with-qlora.html)
9. [LLM From Scratch](docs/generative-ai/llm-from-scratch/index.html)
    - [Part 1: How LLMs Read Text: Tokenization, BPE, and Word Embeddings Explained](docs/generative-ai/llm-from-scratch/00_working-with-text-data.html)
    - [Part 2: Self Attention Explained without Trainable Weights](docs/generative-ai/llm-from-scratch/01_Self_Attention_In_Work_Part_1.html)
    - [Part 3: Attention From Scratch: Self-Attention to Multi-Head Attention](docs/generative-ai/llm-from-scratch/02_Attention_From_Scratch_SelfAttention_to_MultiHead.html)
10. [Generative AI with LangChain](docs/generative-ai/langchain/index.html)
11. [NLP with Hugging Face](docs/generative-ai/nlp-with-huggingface/index.html)
12. [Kubernetes: Ingress & FastAPI Model Deployment](docs/mlops/deploying-fastAPI-models-on-k8s-with-ingress.html)
13. [YOLOv5 Custom Object Detection](docs/computer-vision/yolo-object-detection.html)
14. [Docker Compose and PyTorch Lightning](docs/mlops/docker-compose.html)
15. [Hyperparameter Tuning and Experiment Tracking](docs/mlops/hyperparameter-tuning.html)
16. [Deployment with Gradio](docs/mlops/deployment-with-gradio.html)
17. [Deployment with Litserve](/docs/mlops/deployment-with-litserve.html)

<!-- 
<section class="project-categories-grid">
  <a href="/docs/agentic-ai/index.html" class="category-tile">Agentic AI</a>
  <a href="/docs/computer-vision/index.html" class="category-tile">Computer Vision</a>
  <a href="/docs/generative-ai/index.html" class="category-tile">Generative AI</a>
  <a href="/docs/mlops/index.html" class="category-tile">MLOps</a>
</section> -->


# Certification

<br>

<div class="certification-card">
  <a href="https://learn.microsoft.com/en-us/users/ankitamungalpara-0103/credentials/92d5dc7947b76e29" class="cert-link">
    <img src="https://img-c.udemycdn.com/open-badges/v2/badge-class/1456157181/azure-data-scientist-associate-600x60016151426175978877264.png" 
         alt="Azure Data Scientist Associate Badge" 
         class="cert-logo">
    <div class="cert-info">
      <strong>Microsoft Certified: Azure Data Scientist Associate</strong><br>
    </div>
  </a>
</div>


<div class="my-12 border-t border-gray-200 opacity-70"></div>




# Publication

<br>

<div class="publication-card">
  <a href="/docs/publication/" class="publication-link" target="_blank">
    <img src="https://media.springernature.com/w316/springer-static/cover-hires/book/978-3-031-43759-5?as=webp"
         alt="Springer Logo" class="publication-logo">
    <div class="publication-info">
      <strong>Forest Stand Height Estimation using Inversion of RVoG Model over Forest of North-Eastern India</strong><br>
      <em>Ankita Mungalpara, Dr. Sanid Chirakkal, Dr. Deepak Putrevu, Prof. Suman Mitra</em><br>
      <span>Presented at CAJG 2020</span><br>
      <span>Published in <em>Advances in Science, Technology & Innovation</em>, Springer</span>
    </div>
  </a>
</div>


# Recent Blogs

<br>

<section class="recent-blogs">
  <div class="blog-list">
    <article class="blog-card">
    <a href="docs/agentic-ai/2025-04-05-mcp.html" class="blog-link">
    <img src="https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcb35b6dc-a7e7-42ef-b0a5-8ea42c4ffbc3_959x755.png"
      alt="Model Context Protocol illustration"
      class="blog-image"/>

    <div class="blog-content">
      <h3>Understanding MCP <em>(Model Context Protocol)</em></h3>
      <p class="blog-meta">Published · May 10 2025</p>

      <!-- tag chips -->
      <div class="blog-tags">
        <span class="tag">Agentic AI</span>
        <span class="tag">Model Context Protocol (MCP)</span>
        <span class="tag">Generative AI</span>
      </div>

      <!-- clamped preview -->
      <p class="blog-preview">
        In today's fast-paced AI era, one of the most difficult tasks for developers is seamlessly connecting large language models (LLMs) to the data sources and tools required...
      </p>

      <!-- <span class="read-more">Read more →</span> -->
      <a href="docs/agentic-ai/2025-04-05-mcp.html" class="read-more">Read more →</a>
    </div>
    </a>
    </article>

    <article class="blog-card">
    <a href="docs/generative-ai/2025-04-05-finetune-with-lora.html" class="blog-link">
    <img src="https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F493b62da-91e6-4f6b-896f-4af8c5eb1405_580x514.png"
      alt="Model Context Protocol illustration"
      class="blog-image"/>

    <div class="blog-content">
      <h3>How I Fine-Tuned Mistral-7B Model with LoRA <em>(Low-Rank Adaptation)</em></h3>
      <p class="blog-meta">Published · May 10 2025</p>

      <!-- tag chips -->
      <div class="blog-tags">
        <span class="tag">SFT</span>
        <span class="tag">PEFT</span>
        <span class="tag">Post-Training LLMs</span>
        <span class="tag">Generative AI</span>
      </div>

      <!-- clamped preview -->
      <p class="blog-preview">
        Large Language Models (LLMs) are initially trained on vast, different text corpora scraped from the internet. This pre-training phase teaches them statistical...
      </p>

      <!-- <span class="read-more">Read more →</span> -->
      <a href="docs/generative-ai/2025-04-05-finetune-with-lora.html" class="read-more">Read more →</a>
    </div>
    </a>
    </article>


    <article class="blog-card">
    <a href="docs/generative-ai/2025-02-21-clip-from-scratch.html" class="blog-link">
    <img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*OVi8blLZw_wf2rrxdlfbdg.png"
      alt="Model Context Protocol illustration"
      class="blog-image"/>

    <div class="blog-content">
      <h3>Building CLIP <em>(Contrastive Language–Image Pre-training)</em> From Scratch</h3>
      <p class="blog-meta">Published · May 10 2025</p>

      <!-- tag chips -->
      <div class="blog-tags">
        <span class="tag">CLIP Training</span>
        <span class="tag">Multi-Head Attention</span>
        <span class="tag">Positional Embedding</span>
      </div>

      <!-- clamped preview -->
      <p class="blog-preview">
        Contrastive Language-Image Pre-training (CLIP) was developed by OpenAI and first introduced in the paper “Learning Transferable Visual Models From Natural...
      </p>

      <!-- <span class="read-more">Read more →</span> -->
      <a href="docs/generative-ai/2025-02-21-clip-from-scratch.html" class="read-more">Read more →</a>
    </div>
    </a>
    </article>

  </div>
</section>



Feel free to connect or explore more on my [GitHub](https://github.com/AnkitaMungalpara) or [LinkedIn](https://www.linkedin.com/in/ankita-mungalpara).


