---
# layout: minimal
title: Conect with me
nav_order: 11
has_toc: false
---
<!-- 
# Connect with Me

<a href="https://github.com/ankitapatel" target="_blank">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub" width="40" height="40" style="margin-right: 10px;" />
</a> -->

# Connect with Me
{: .no_toc }

Feel free to reach out—I’d love to hear from you!

<form id="contact-form"
  action="https://formspree.io/f/xwpbdnjk" 
  method="POST" 
  onsubmit="setTimeout(() => document.getElementById('contact-form').reset(), 10)"
  style="max-width: 600px;">

  <label for="name"><strong>Name</strong></label><br/>
  <input type="text" id="name" name="name" required style="width:100%; padding:8px; margin-bottom:12px;"/>

  <label for="email"><strong>Email</strong></label><br/>
  <input type="email" id="email" name="email" required style="width:100%; padding:8px; margin-bottom:12px;"/>

  <label for="subject"><strong>Subject</strong></label><br/>
  <input type="text" id="subject" name="subject" required style="width:100%; padding:8px; margin-bottom:12px;"/>

  <label for="message"><strong>Message</strong></label><br/>
  <textarea id="message" name="message" rows="6" required style="width:100%; padding:8px; margin-bottom:12px;"></textarea>

  <button type="submit" style="padding:10px 20px; background-color:#000; color:#fff; border:none; cursor:pointer;">
    Send Message
  </button>
</form> 
