"use client";

import * as React from "react";
import {
  LayoutDashboardIcon,
  BrainCircuitIcon,
  BarChart3Icon,
  BookOpenIcon,
  TagsIcon,
  SearchIcon,
  GithubIcon,
} from "lucide-react";

import { NavMain } from "@/components/nav-main";
import { NavSecondary } from "@/components/nav-secondary";
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";

const data = {
  navMain: [
    {
      title: "Dashboard",
      url: "/dashboard",
      icon: <LayoutDashboardIcon />,
    },
    {
      title: "Classifier",
      url: "/classify",
      icon: <BrainCircuitIcon />,
    },
    {
      title: "Model Comparison",
      url: "/models",
      icon: <BarChart3Icon />,
    },
    {
      title: "Bibliometrics",
      url: "/bibliometrics",
      icon: <BookOpenIcon />,
    },
    {
      title: "Topic Explorer",
      url: "/topics",
      icon: <TagsIcon />,
    },
    {
      title: "Similar Papers",
      url: "/similar",
      icon: <SearchIcon />,
    },
  ],
  navSecondary: [
    {
      title: "GitHub",
      url: "https://github.com/bastab00/final-year",
      icon: <GithubIcon />,
    },
  ],
};

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  return (
    <Sidebar collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              className="data-[slot=sidebar-menu-button]:p-1.5!"
              render={<a href="/dashboard" />}
            >
              <BrainCircuitIcon className="size-5!" />
              <span className="text-base font-semibold">InflationAI</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.navMain} />
        <NavSecondary items={data.navSecondary} className="mt-auto" />
      </SidebarContent>
    </Sidebar>
  );
}
