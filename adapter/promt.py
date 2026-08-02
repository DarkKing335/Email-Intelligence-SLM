def extract_email_metadata(email_text):
    system_prompt = "You are a precise data extraction assistant. You must output ONLY a valid JSON object."
    
    # Bổ sung ép chuẩn Schema cho entities và các quy tắc phụ cho draft
    user_prompt = f"""Analyze the following email and extract its metadata into a strict JSON format with exactly these keys:
- "classification": You MUST choose exactly ONE label based on these strict rules:
    * "action_required": Use ONLY for system alerts, build/pipeline failures, or tasks assigned by a professor.
    * "respond": Use ONLY for personal communication, meeting requests, or collaborations that require a human reply.
    * "notification": Use for informational updates (e.g., successful training jobs, patch notes) needing no immediate action.
    * "social": Use ONLY for casual check-ins, like gym/workout schedules or diet plans.
    * "spam": Use ONLY for unsolicited promotions, sales, or discounts.

- "priority": Choose strictly based on these rules:
    * "high": Critical system failures, pipeline breaks, or urgent professor requests.
    * "medium": Normal meeting requests, training job completions, or standard notifications.
    * "low": Social chats, gym plans, game newsletters, or spam.
    
- "summary": (a short summary)

- "entities": A strict JSON object. YOU MUST ONLY USE THE FOLLOWING KEYS. If an entity is not present, omit the key completely. DO NOT invent new keys or arrays.
    * "people": (array of strings, e.g., ["Linh", "Casey"])
    * "date": (string, format YYYY-MM-DD or as written in text)
    * "time": (string)
    * "project_or_product": (array of strings, e.g., ["SUMO Urban Traffic Model", "Genshin Impact"])
    * "technology": (array of strings, e.g., ["C++ pipeline", "PostgreSQL", "Docker"])
    * "misc_items": (array of strings, for other specific items like ["chicken breast", "50% discount", "fullscreen configuration"])

- "recommended_action": (short string. Must be "Ignore the offer." for spam, or empty string "" for patch notes/newsletters).

- "draft": (a short reply if needed. MUST be empty string "" if classification is "notification", "spam", or "action_required" for system builds).

Email to analyze:
{email_text}

JSON Output:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=512, 
        temperature=0.1,  
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response