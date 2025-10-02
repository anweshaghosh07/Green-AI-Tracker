#!/usr/bin/env python3
"""
Automated demo recording with Playwright:
- Opens the Streamlit dashboard
- Selects a model filter
- Navigates to 'Insights' tab
- Records a short video (15–20s)
"""

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"   # Change if running on EC2/public URL

def main():
    with sync_playwright() as p:
        # Enable video recording
        browser = p.chromium.launch(headless=False, slow_mo=500)  # headless=False so charts animate
        context = browser.new_context(
            viewport={"width":1280,"height":800},
            record_video_dir="tests/screenshots/",   # output folder
            record_video_size={"width":1280, "height":800}
        )

        page = context.new_page()

        # Open dashboard
        page.goto(URL, timeout=60000)
        page.wait_for_timeout(4000)  # let dashboard render

        # --- Step 1: Open the Model dropdown ---
        try:
            # Find the first combobox (adjust if you have multiple dropdowns)
            page.get_by_role("combobox").click()
            page.wait_for_timeout(1000)
            
            # Now click the option inside the dropdown
            page.get_by_role("Select Model(s)", name="logreg").click()
            print("✅ Selected LogisticRegression filter")
        except Exception as e:
            print("⚠️ Could not select model filter:", e)

        page.wait_for_timeout(2000)

        # --- Step 2: Navigate to Insights tab ---
        try:
            page.get_by_role("tab", name="Insights").click()
        except Exception:
            page.get_by_text("Insights").click()

        page.wait_for_timeout(10000)  # stay on Insights for 10s

        # Close context → saves video automatically
        context.close()
        browser.close()

        print("🎥 Demo recording saved in tests/screenshots/")

if __name__ == "__main__":
    main()