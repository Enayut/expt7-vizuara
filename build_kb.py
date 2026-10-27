"""Download the public help-center pages (same sources as Experiment 6) into support_faq.txt."""
import re
import requests
from bs4 import BeautifulSoup

SOURCE_URLS = {
    "Order Tracking": [
        "https://www.shipaid.com/blog/how-do-i-track-my-order-on-shopify-a-comprehensive-guide-for-e-commerce-merchants",
        "https://www.eesel.ai/blog/ecommerce-faq-examples",
    ],
    "Returns and Refunds": [
        "https://usource.me/top-20-most-commonly-asked-ecommerce-customer-service-queries/",
    ],
    "Account Access": [
        "https://support.microsoft.com/en-us/help/4026971/microsoft-account-how-to-reset-your-password",
        "https://www.fittingbox.com/en/resources/help-center/how-do-i-change-my-password",
    ],
    "Billing and Payments": [
        "https://help.nordlayer.com/docs/i-have-been-charged-twice.md",
        "https://help.letsdothis.com/en/articles/10546731-what-to-do-if-you-have-been-charged-twice",
        "https://help.cathopic.com/help-troubleshooting/i-was-charged-twice/",
    ],
    "Technical Support": [
        "https://support.mozilla.org/en-US/kb/websites-dont-load-troubleshoot-and-fix-errors",
    ],
}
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
MAX_CHARS_PER_PAGE = 6000

FALLBACK_FAQ = """Order Tracking
Customers can view shipment updates in the Orders section of their account. A tracking link is shown after an order is dispatched. If tracking has not updated for 48 hours, contact support with the order reference. The assistant cannot access live order status.

Returns and Refunds
Unused items in their original packaging may be requested for return within 14 days of delivery. Some items may be excluded. To start a return, open Orders, choose the item, and select Return Request. After an approved return is received, a refund is usually processed to the original payment method within 5 to 7 business days. The assistant cannot approve a refund.

Account Access
If you cannot sign in, use the Forgot Password option on the sign-in page. Check your email for the password reset link. If the email does not arrive, check the spam folder and confirm that you entered the registered email address. Never share your password or one-time verification code.

Billing and Payments
Payment status is shown in the order details. If a payment appears to have been charged twice, do not submit repeated payments. Contact the billing support team with the transaction date and order reference. Do not share full card numbers or security codes.

Technical Support
If the website is not loading, refresh the page, check your internet connection, and try an updated browser. If the problem continues, clear the browser cache or try a private window. When contacting support, describe the error message and the steps that led to it. Do not share passwords or verification codes.

Contact Support
For an issue that is not covered by these FAQs, contact the support team through the Help section of the website. Include a brief description and relevant order reference, but do not send passwords, one-time codes, or full payment-card details.
"""


def fetch_page_text(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    if url.endswith(".md") or "text/plain" in r.headers.get("Content-Type", ""):
        text = r.text
    else:
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()
        main = soup.find("article") or soup.find("main") or soup.body or soup
        text = main.get_text("\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
    return text[:MAX_CHARS_PER_PAGE]


if __name__ == "__main__":
    pages = []
    for topic, urls in SOURCE_URLS.items():
        for url in urls:
            try:
                text = fetch_page_text(url)
                if len(text) < 300:
                    raise ValueError("page text too short")
                pages.append((topic, url, text))
                print(f"[OK]   {topic:22s} {len(text):5d} chars  {url}")
            except Exception as e:
                print(f"[FAIL] {topic:22s} {url} -> {e}")
    with open("support_faq.txt", "w", encoding="utf-8") as f:
        for topic, url, text in pages:
            f.write(f"=====PAGE=====\n{topic}\nSource: {url}\n{text}\n\n")
        # always add the manual's sample FAQ so every topic (e.g. Technical Support) is covered
        f.write(f"=====PAGE=====\nBuilt-in FAQ\nSource: built-in\n{FALLBACK_FAQ}")
    print(f"support_faq.txt written ({len(pages)} pages)")
