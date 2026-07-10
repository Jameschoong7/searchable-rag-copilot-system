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

type DemoProfile = {
  user: string;
  role: string;
  department: string;
  label: string;
};

const DEFAULT_PROFILE: DemoProfile = {
  user: "teams_employee_hr",
  role: "General Employee",
  department: "HR",
  label: "General Employee / HR",
};

const DEMO_PROFILES: Record<string, DemoProfile> = {
  "/use-hr": DEFAULT_PROFILE,
  "/use-it-manager": {
    user: "teams_pm_it",
    role: "Project Manager",
    department: "IT",
    label: "Project Manager / IT",
  },
  "/use-admin": {
    user: "teams_admin_jc",
    role: "System Admin",
    department: "IT",
    label: "System Admin / IT",
  },
};

const conversationProfiles = new Map<string, DemoProfile>();

function getConversationProfile(conversationId: string): DemoProfile {
  return conversationProfiles.get(conversationId) || DEFAULT_PROFILE;
}

function formatProfileHelp(profile: DemoProfile): string {
  return [
    `Current access profile: ${profile.label}.`,
    "",
    "Available demo commands:",
    "- /profile",
    "- /use-hr",
    "- /use-it-manager",
    "- /use-admin",
  ].join("\n");
}

app.on("install.add", async ({ send }) => {
  await send(
    [
      "Searchable RAG Copilot is connected.",
      "",
      `Current access profile: ${DEFAULT_PROFILE.label}.`,
      "You can ask HR knowledge-base questions here. Access to restricted department documents is controlled by the shared backend.",
      "",
      "Use /profile to view or change the current demo profile.",
    ].join("\n")
  );
});

app.on("message", async ({ send, activity }) => {
  const question = activity.text?.trim();

  if (!question) {
    await send("Please enter a question.");
    return;
  }

  const conversationId = activity.conversation.id;
  const command = question.toLowerCase();

  if (command === "/profile") {
    await send(formatProfileHelp(getConversationProfile(conversationId)));
    return;
  }

  if (DEMO_PROFILES[command]) {
    conversationProfiles.set(conversationId, DEMO_PROFILES[command]);
    await send(`Switched access profile to ${DEMO_PROFILES[command].label}.`);
    return;
  }

  const profile = getConversationProfile(conversationId);

  await send({ type: "typing" });

  const apiBaseUrl = process.env.RAG_API_BASE_URL || "http://127.0.0.1:8000";

  try {
    const submitResponse = await fetch(`${apiBaseUrl}/chat/jobs`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        role: profile.role,
        department: profile.department,
        user: profile.user,
        session_id: undefined,
        use_memory: true,
        client: "teams",
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
  } catch (error) {
    console.error("RAG backend request failed", error);
    await send(
      "The knowledge backend is not reachable. Please make sure the FastAPI service is running."
    );
  }
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
