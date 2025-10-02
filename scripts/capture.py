#!/usr/bin/env python3
"""
Automated dashboard capture with Playwright:
- Opens Streamlit dashboard
- Selects a model filter
- Takes screenshot of main dashboard
- Clicks on 'Insights' tab
- Takes screenshot of insights page
"""

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"  # or your EC2/public URL

def main():
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)  # set headless=False to watch
        page = browser.new_page(viewport={"width":1280,"height":900})

        # Open dashboard
        page.goto(URL, timeout=60000)
        page.wait_for_timeout(5000)  # wait for Streamlit to finish rendering

        # --- Step 1: Select model filter (example: LogisticRegression) ---
        # Assumes the filter uses a label or visible text "Model"
        try:
            # Streamlit sidebar usually has label text -> locate via text match
            page.get_by_label("Model").select_option("LogisticRegression")
        except Exception as e:
            print("Could not select model filter, check locator:", e)

        # Screenshot after filter applied
        page.screenshot(path="tests/screenshots/dashboard_filtered.png", full_page=True)
        print("Saved dashboard_filtered.png")

        # --- Step 2: Click 'Insights' tab ---
        try:
            page.get_by_role("tab", name="Insights").click()
        except Exception:
            # Fallback: click via text
            page.get_by_text("Insights").click()

        page.wait_for_timeout(3000)  # let charts load

        # Screenshot of insights tab
        page.screenshot(path="tests/screenshots/dashboard_insights.png", full_page=True)
        print("Saved dashboard_insights.png")

        browser.close()

if __name__ == "__main__":
    main()