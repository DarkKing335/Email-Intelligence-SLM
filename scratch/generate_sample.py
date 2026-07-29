import csv
import random
from pathlib import Path

# Raw categories matching configs/label_taxonomy.yaml mapping keys
CATEGORIES = ["spam", "promotions", "verify_code", "updates", "forum", "social_media", "primary", "billing", "support", "notifications"]
DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "company.com", "bank.com", "service.io", "forum.org", "social.net"]
NAMES = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Hannah"]

SUBJECTS = {
    "spam": [
        "Earn $$$ from home now!",
        "Special offer: custom replicas inside",
        "Re: Wire transfer confirmation needed immediately",
        "Get rich quick with this one simple trick",
        "Exclusive deals on weight loss pills!"
    ],
    "promotions": [
        "Summer Sale: 50% off all items!",
        "Weekly Newsletter - Tips for clean coding",
        "Your weekly digest from ShopCo",
        "Upgrade your subscription today and save",
        "New arrivals are here! Check them out."
    ],
    "verify_code": [
        "Your verification code is 482910",
        "One-time password (OTP) for account login",
        "Action required: Confirm your registration",
        "Security Alert: New sign-in detected",
        "Confirm your email address to continue"
    ],
    "updates": [
        "Project status update: Sprint 4 complete",
        "Meeting notes from today's sync",
        "Change in company policy regarding remote work",
        "Upcoming system maintenance notice",
        "Shared document update: Q3 Roadmap"
    ],
    "forum": [
        "[Python Forum] New reply to 'How to parse JSON in Python?'",
        "[AI community] Weekly discussion topic: SLM fine-tuning",
        "New thread: Best practices for database migration",
        "Your post was upvoted in developer-hub",
        "Digest for r/programming: weekly highlights"
    ],
    "social_media": [
        "John Doe sent you a friend request",
        "Jane Doe mentioned you in a comment",
        "New connection request on LinkUp",
        "Trending in your network: AI trends",
        "You have 3 new notifications from TwitBook"
    ],
    "primary": [
        "Quick question about the proposal",
        "Can we reschedule our sync to 3pm?",
        "Draft review for the client contract",
        "Feedback on the UI design mockup",
        "Urgent: client needs updates on the dashboard"
    ],
    "billing": [
        "Invoice INV-2026-0048 for July services",
        "Your payment has been received",
        "Action required: Update your payment method",
        "Receipt for your subscription renewal",
        "Your monthly statement is ready"
    ],
    "support": [
        "Ticket #48291: Login issues resolved",
        "Support request: Unable to upload attachments",
        "Feedback request regarding your support experience",
        "Re: How to reset my password?",
        "We are looking into your issue"
    ],
    "notifications": [
        "Build success: version 1.2.0-rc3",
        "Alert: CPU usage exceeded 90% on Server 3",
        "Deploying to staging environment",
        "Backup completed successfully",
        "Log rotation triggered on host-db-1"
    ]
}

BODIES = {
    "spam": [
        "Hi! Buy now and receive a 90% discount. Limited slots available. Click here: http://get-rich-fast.scam/win. Phone: +1-800-555-0199. SSN: 000-12-3456.",
        "Dear Friend, you have been selected for a special cash prize. Please send your credit card details: 4111-2222-3333-4444 to claim. Sincerely, Dr. Scam.",
        "Get your weight loss pills today! Safe, natural, fast. Free shipping on all orders. Regards, Sales team.",
        "URGENT: Your account has been compromised! Please verify your identity immediately by calling +1-888-555-9999 or email us at threat-alert@insecure.net."
    ],
    "promotions": [
        "Hello from ShopCo! Check out our summer collection. Use code SUMMER50 at checkout for half off. Unsubscribe at http://shopco.com/unsubscribe.",
        "Here is your weekly digest of programming tutorials. Learn Python, Rust, and Go. Thanks for subscribing!",
        "Don't miss out on our limited time offer! Get 3 months of premium service for the price of 1. Click here to learn more.",
        "Weekly newsletter: How to build responsive layouts using clean CSS grids. Regards, The Dev Team."
    ],
    "verify_code": [
        "Your verification code is: 958204. This code is valid for 10 minutes. Do not share it with anyone. IP address: 192.168.1.50.",
        "Your login code is 8847. Enter it immediately to access your account. If you did not request this, please contact security@bank.com.",
        "Confirm your email address by entering this OTP: 552918 on our website. Best regards, Security Team.",
        "Security Alert: A login attempt was detected from a new location. Confirm it was you by entering code 103948."
    ],
    "updates": [
        "Hi Team, here is the status of Sprint 4. All tickets have been merged. Review notes at http://wiki.company.internal/sprint-4. Kind regards, Alice.",
        "Hello all, please note system maintenance is scheduled this Sunday from 2 AM to 4 AM UTC. Expect minor outages. Regards, IT Support.",
        "The Q3 roadmap has been updated. Please review the goals and add comments by end of day Friday. Thanks, Product Team.",
        "Please find the remote work policy update document attached. All employees should read and sign the acknowledgment."
    ],
    "forum": [
        "You subscribed to updates for 'JSON parsing'. Bob replied: 'You can use the built-in json module.' View thread at http://forum.dev/topic/48.",
        "Weekly digest: 5 discussions you might have missed in the Rust community. Read more on developer-hub.org.",
        "Charlie posted a new topic: 'Best practices for writing unit tests'. Reply to join the discussion.",
        "Your post 'How to train SLMs' was upvoted 25 times today. Keep up the good work!"
    ],
    "social_media": [
        "Hi! John Doe sent you a friend request. View profile: http://social.net/johndoe. You can turn off these notifications in settings.",
        "Jane Doe mentioned you in a comment: 'Great job on the presentation! @developer'. Click to reply.",
        "Emma wants to connect with you on LinkUp. 'I'd like to add you to my professional network.' Regards, LinkUp.",
        "You have 3 new notifications from TwitBook. David posted a new photo, Hannah liked your post, and 1 other event."
    ],
    "primary": [
        "Hi Bob, I reviewed the draft contract. Please check section 4.2 regarding liabilities and let me know if we need adjustments. Thanks, Alice.",
        "Can we move our daily sync to 3 PM today? I have a client call at 10 AM. Let me know if that works. Regards, Charlie.",
        "Here is the design mockup for the main landing page. Let me know your thoughts on the color palette. Sincerely, David.",
        "Urgent: The client found a bug in the reporting dashboard. The values are not loading correctly. Can you inspect immediately?"
    ],
    "billing": [
        "Invoice INV-2026-0048 for software consulting services is ready. Total due: $2,500.00. Payment due by 2026-08-15. Regards, Finance.",
        "Thank you for your payment of $49.99 for your monthly subscription. Your receipt is attached. Billing support: billing@service.io.",
        "Action Required: Your credit card on file (ending in 4321) expired. Please update billing details immediately to avoid disruption.",
        "Your monthly statement for account 84920 is now available. Log in to your portal to download. Thanks, Bank Corp."
    ],
    "support": [
        "Hello, your ticket #48291 regarding login issues has been updated. We have reset your session. Please try again. Best, Helpdesk.",
        "Dear Customer, we received your support request about uploading attachments. Our technical team is reviewing it. ticket-id: 9382.",
        "We would love to get your feedback on your recent support interaction. Please take a 1-minute survey. Thanks, Customer Care.",
        "Hi, I reset your password as requested. Please use the temporary link: http://service.io/reset-pwd. Regards, Support."
    ],
    "notifications": [
        "Build success: pipeline run #382 completed in 4 mins 12 secs. All tests passed. Version: v1.2.0-rc3.",
        "CRITICAL ALERT: CPU utilization on server-db-primary is at 94.2%. Triggering auto-scale. Timestamp: 2026-07-29T12:00:00Z.",
        "Deploying commit e82a9f to staging environment. Status: In-progress.",
        "Automated backup of database completed successfully. File: backup_20260729.tar.gz. Size: 14.2 GB.",
        "Log rotation triggered successfully. Compressed logs moved to archive storage."
    ]
}

def generate_sample_dataset(output_path: Path, count: int = 100):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sender", "subject", "body", "category"])
        
        for i in range(count):
            # Pick a category
            cat = random.choice(CATEGORIES)
            
            # Generate sender
            sender_name = random.choice(NAMES)
            sender_domain = random.choice(DOMAINS)
            sender = f"{sender_name} <{sender_name.lower()}@{sender_domain}>"
            
            # Generate subject
            subj_list = SUBJECTS.get(cat, ["Default Subject"])
            subject = random.choice(subj_list)
            # Add some variation to make them unique
            if random.random() < 0.3:
                subject = f"Fwd: {subject}"
            elif random.random() < 0.3:
                subject = f"Re: {subject}"
                
            # Generate body
            body_list = BODIES.get(cat, ["Default email body text goes here."])
            body = random.choice(body_list)
            # Add some unique text or code to body to prevent exact duplicates naturally
            body = f"{body} [Ref: {random.randint(10000, 99999)}]"

            writer.writerow([sender, subject, body, cat])

if __name__ == "__main__":
    random.seed(42)
    # Generate 100 sample records to emails_sample.csv
    csv_path = Path("e:/FPT/Email Intelligence SLM/data/samples/emails_sample.csv")
    generate_sample_dataset(csv_path, 100)
    print(f"Generated 100 sample rows at {csv_path}")
