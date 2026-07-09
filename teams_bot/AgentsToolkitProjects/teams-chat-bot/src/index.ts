import { ManagedIdentityCredential } from "@azure/identity";
import { cardAttachment, TokenCredentials } from "@microsoft/teams.api";
import { App } from "@microsoft/teams.apps";
import { IAdaptiveCard } from "@microsoft/teams.cards";
import { ConsoleLogger } from "@microsoft/teams.common/logging";
import { DevtoolsPlugin } from "@microsoft/teams.dev";

import {
  createCard,
  createConversationMembersCard,
  createDummyCards,
  createLinkUnfurlCard,
  createMessageDetailsCard,
} from "./card";

const createTokenFactory = () => {
  return async (scope: string | string[], tenantId?: string): Promise<string> => {
    const managedIdentityCredential = new ManagedIdentityCredential({
      clientId: process.env.CLIENT_ID,
    });
    const scopes = Array.isArray(scope) ? scope : [scope];
    const tokenResponse = await managedIdentityCredential.getToken(scopes, {
      tenantId: tenantId,
    });

    return tokenResponse.token;
  };
};

// Configure authentication using TokenCredentials
const tokenCredentials: TokenCredentials = {
  clientId: process.env.CLIENT_ID || "",
  token: createTokenFactory(),
};

// Use managed identity in cloud environment, otherwise use devtools plugin for local development
const options =
  process.env.BOT_TYPE === "UserAssignedMsi"
    ? { ...tokenCredentials }
    : { plugins: [new DevtoolsPlugin()] };

const app = new App({
  ...options,
  logger: new ConsoleLogger("teams-chat-bot", { level: "debug" }),
  skipAuth: !process.env.CLIENT_ID,
});

app.on("install.add", async ({ send }) => {
  await send(
    [
      "Searchable RAG Copilot is connected.",
      "",
      "Current access profile: General Employee / HR.",
      "You can ask HR knowledge-base questions here. Access to restricted department documents is controlled by the shared backend.",
    ].join("\n")
  );
});

app.on("message", async ({ send, activity }) => {
  const question = activity.text?.trim();

  if (!question) {
    await send("Please enter a question.");
    return;
  }

  await send({ type: "typing" });

  const apiBaseUrl = process.env.RAG_API_BASE_URL || "http://127.0.0.1:8000";

  const submitResponse = await fetch(`${apiBaseUrl}/chat/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      role: "General Employee",
      department: "HR",
      user: "teams_employee_hr",
      session_id: undefined,
      use_memory: true,
    }),
  });

  if (!submitResponse.ok) {
    const errorText = await submitResponse.text();
    await send(`Could not submit question to RAG backend: ${errorText}`);
    return;
  }

  const submittedJob = await submitResponse.json();
  const jobId = submittedJob.job_id;

  for (let attempt = 0; attempt < 30; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 1000));

    const jobResponse = await fetch(`${apiBaseUrl}/admin/jobs/${jobId}`);

    if (!jobResponse.ok) {
      const errorText = await jobResponse.text();
      await send(`Could not check RAG job status: ${errorText}`);
      return;
    }

    const job = await jobResponse.json();

    if (job.status === "succeeded") {
      const result = job.result;
      const sources = result.sources || [];

      const answerText = result.answer || "No answer returned.";
      const sourceText =
        sources.length > 0
          ? `\n\nSources:\n${sources.map((source: string) => `- ${source}`).join("\n")}`
          : "";

      await send(`${answerText}${sourceText}`);

      return;
    }

    if (job.status === "failed") {
      await send(job.message || "The RAG backend failed to answer.");
      return;
    }
  }

  await send("The RAG backend is still processing. Please try again shortly.");
});

// :snippet-start: message-ext-query-link
app.on("message.ext.query-link", async ({ activity }) => {
  const { url } = activity.value;

  if (!url) {
    return { status: 400 };
  }

  const { card, thumbnail } = createLinkUnfurlCard(url);
  const attachment = {
    ...cardAttachment("adaptive", card), // expanded card in the compose box...
    preview: cardAttachment("thumbnail", thumbnail), //preview card in the compose box...
  };

  return {
    composeExtension: {
      type: "result",
      attachmentLayout: "list",
      attachments: [attachment],
    },
  };
});
// :snippet-end: message-ext-query-link
// :snippet-start: message-ext-submit
app.on("message.ext.submit", async ({ activity }) => {
  const { commandId } = activity.value;
  let card: IAdaptiveCard;

  if (commandId === "createCard") {
    // activity.value.commandContext == "compose"
    card = createCard(activity.value.data);
  } else if (commandId === "getMessageDetails" && activity.value.messagePayload) {
    // activity.value.commandContext == "message"
    card = createMessageDetailsCard(activity.value.messagePayload);
  } else {
    throw new Error(`Unknown commandId: ${commandId}`);
  }

  return {
    composeExtension: {
      type: "result",
      attachmentLayout: "list",
      attachments: [cardAttachment("adaptive", card)],
    },
  };
});
// :snippet-end: message-ext-submit

// :snippet-start: message-ext-open
app.on("message.ext.open", async ({ activity, api }) => {
  const conversationId = activity.conversation.id;
  const members = await api.conversations.members(conversationId).get();
  const card = createConversationMembersCard(members);

  return {
    task: {
      type: "continue",
      value: {
        title: "Conversation members",
        height: "small",
        width: "small",
        card: cardAttachment("adaptive", card),
      },
    },
  };
});
// :snippet-end: message-ext-open

// :snippet-start: message-ext-query
app.on("message.ext.query", async ({ activity }) => {
  const { commandId } = activity.value;
  const searchQuery = activity.value.parameters![0].value;

  if (commandId == "searchQuery") {
    const cards = await createDummyCards(searchQuery);
    const attachments = cards.map(({ card, thumbnail }) => {
      return {
        ...cardAttachment("adaptive", card), // expanded card in the compose box...
        preview: cardAttachment("thumbnail", thumbnail), // preview card in the compose box...
      };
    });

    return {
      composeExtension: {
        type: "result",
        attachmentLayout: "list",
        attachments: attachments,
      },
    };
  }

  return { status: 400 };
});
// :snippet-end: message-ext-query

(async () => {
  await app.start();
})();
