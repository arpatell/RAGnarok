import * as cheerio from "cheerio";

interface LoginDetectionInput {
  finalUrl: string;
  status: number;
  html: string;
  panelCount: number;
  treatLowPanelCountAsLoginWall?: boolean;
}

const LOGIN_URL_PATTERN = /\/(login|signin|register)(\/|$|\?)/i;
const HARD_GATE_TEXT = /(login|sign in|register|create account).{0,40}(to read|to continue|to view)/i;

export function isLikelyLoginWall(input: LoginDetectionInput): boolean {
  if (input.status === 401 || input.status === 403) {
    return true;
  }

  if (LOGIN_URL_PATTERN.test(input.finalUrl)) {
    return true;
  }

  const $ = cheerio.load(input.html);
  const hasLoginForm =
    $("form input[type='password']").length > 0 ||
    $("form input[name*='email' i]").length > 0 ||
    $("form input[name*='login' i]").length > 0;

  if (input.treatLowPanelCountAsLoginWall !== false && input.panelCount < 2) {
    return true;
  }

  if (hasLoginForm) {
    const bodyText = $("body").text().replace(/\s+/g, " ").trim();
    if (HARD_GATE_TEXT.test(bodyText) && input.panelCount < 5) {
      return true;
    }
  }

  return false;
}
