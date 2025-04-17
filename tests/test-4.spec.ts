import { test, expect } from '@playwright/test';

test('test', async ({ page }) => {
  await page.goto('https://flybase.org/');
  await page.waitForLoadState('networkidle');   // ensure page is fully rendered
  await page.getByRole('link', { name: 'GAL4 etc' }).click();
  await page.getByRole('textbox', { name: 'e.g., third instar larval' }).click();
  await page.getByRole('textbox', { name: 'e.g., third instar larval' }).fill('th');
  await page.getByText('early third instar larval').click();
  await page.getByRole('button', { name: 'Search' }).click();
});