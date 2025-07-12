// utils/api.js

export const callBackend = async ({ url, method = 'POST', payload = {} }) => {
  try {
    const response = await fetch(url, {
      method,
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Error ${response.status}: ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Backend call failed:", error);
    return { error: error.message };
  }
};
