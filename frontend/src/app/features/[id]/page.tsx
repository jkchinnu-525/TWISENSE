import Feature1 from "@/app/features/[id]/feature1";
import Feature2 from "./feature2";
import { notFound } from "next/navigation";

export default async function FeaturesPage({
  params,
}: {
  params: { id: string };
}) {
  const { id } = await params;
  console.log("Feature ID:", id);
  if (id === "1") {
    return <Feature1 />;
  }
  if (id === "2") {
    return <Feature2 />;
  }
  notFound();
}
